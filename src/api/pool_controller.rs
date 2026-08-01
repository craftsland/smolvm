//! Background reconciliation for automatic held-fork worker pools.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;

use crate::api::handlers::machines::{delete_one, fork_machine_inner};
use crate::api::state::ApiState;
use crate::api::types::ForkRequest;
use crate::pool::{ForkPoolRecord, ForkPoolSlotState};

const RECONCILE_INTERVAL: Duration = Duration::from_secs(1);
const MAX_PROVISIONS_PER_POOL_TICK: usize = 4;

/// Maintains each pool's clean-worker target and reaps finished leases.
pub struct ForkPoolController {
    state: Arc<ApiState>,
    shutdown_rx: watch::Receiver<bool>,
    fills: tokio::task::JoinSet<String>,
    filling: std::collections::HashSet<String>,
}

impl ForkPoolController {
    /// Create a controller sharing the API's durable state and shutdown signal.
    pub fn new(state: Arc<ApiState>, shutdown_rx: watch::Receiver<bool>) -> Self {
        Self {
            state,
            shutdown_rx,
            fills: tokio::task::JoinSet::new(),
            filling: std::collections::HashSet::new(),
        }
    }

    /// Reconcile until server shutdown.
    pub async fn run(mut self) {
        // A provisioning row can only have an in-flight creator in this process.
        // On startup every such row is therefore crash residue: recover a fully
        // booted held VM, otherwise retire it before admitting new capacity.
        if let Err(error) = self.recover_interrupted_provisioning().await {
            tracing::warn!(%error, "failed to recover interrupted fork-pool provisioning");
        }

        let mut ticker = tokio::time::interval(RECONCILE_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        tracing::info!("fork pool controller started");
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    self.reap_fill_tasks();
                    if let Err(error) = self.reconcile_once().await {
                        tracing::warn!(%error, "fork pool reconciliation failed");
                    }
                }
                changed = self.shutdown_rx.changed() => {
                    if changed.is_err() || *self.shutdown_rx.borrow() {
                        self.fills.abort_all();
                        tracing::info!("fork pool controller shutting down");
                        break;
                    }
                }
            }
        }
    }

    fn reap_fill_tasks(&mut self) {
        while let Some(result) = self.fills.try_join_next() {
            match result {
                Ok(pool_name) => {
                    self.filling.remove(&pool_name);
                }
                Err(error) => {
                    tracing::warn!(%error, "fork pool fill task failed");
                    // A panic loses the task's return value, so conservatively
                    // allow every pool to be scheduled again. Slot reservations
                    // still prevent overfill even if another task is winding down.
                    self.filling.clear();
                }
            }
        }
    }

    async fn recover_interrupted_provisioning(&self) -> Result<(), String> {
        let db = self.state.db().clone();
        let pools = tokio::task::spawn_blocking(move || db.list_fork_pools())
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        for pool in pools {
            let db = self.state.db().clone();
            let pool_name = pool.name.clone();
            let slots = tokio::task::spawn_blocking(move || db.list_fork_pool_slots(&pool_name))
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            for slot in slots
                .into_iter()
                .filter(|slot| slot.state == ForkPoolSlotState::Provisioning)
            {
                let db = self.state.db().clone();
                let machine = slot.machine_name.clone();
                let vm = tokio::task::spawn_blocking(move || db.get_vm(&machine))
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                let recoverable = vm
                    .as_ref()
                    .map(|record| {
                        record.forkpoint_held
                            && record.golden.as_deref() == Some(pool.golden.as_str())
                            && record.is_process_alive()
                    })
                    .unwrap_or(false);
                let db = self.state.db().clone();
                let machine = slot.machine_name;
                tokio::task::spawn_blocking(move || {
                    if recoverable {
                        db.mark_fork_pool_slot_ready(&machine, crate::util::current_timestamp())
                    } else {
                        db.mark_fork_pool_slot_retiring(
                            &machine,
                            crate::util::current_timestamp(),
                            Some("controller restarted during worker provisioning".into()),
                        )
                    }
                })
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    async fn reconcile_once(&mut self) -> Result<(), String> {
        let now = crate::util::current_timestamp();
        let db = self.state.db().clone();
        let expired = tokio::task::spawn_blocking(move || db.expire_fork_leases(now))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        for lease in expired {
            tracing::info!(
                pool = %lease.pool_name,
                lease = %lease.id,
                machine = %lease.machine_name,
                "fork pool lease expired"
            );
        }

        self.retire_invalid_ready_workers().await?;
        self.retire_dead_leased_workers().await?;
        self.delete_retired_workers().await?;

        let db = self.state.db().clone();
        tokio::task::spawn_blocking(move || db.finalize_deleted_fork_pools())
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;

        let db = self.state.db().clone();
        let pools = tokio::task::spawn_blocking(move || db.list_fork_pools())
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        for pool in pools.into_iter().filter(|pool| !pool.deleting) {
            if self.filling.contains(&pool.name) {
                continue;
            }
            let db = self.state.db().clone();
            let pool_for_deficit = pool.name.clone();
            let deficit =
                tokio::task::spawn_blocking(move || db.fork_pool_ready_deficit(&pool_for_deficit))
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
            if deficit > 0 && self.filling.insert(pool.name.clone()) {
                let state = self.state.clone();
                let pool_name = pool.name.clone();
                self.fills.spawn(async move {
                    Self::fill_pool(state, pool).await;
                    pool_name
                });
            }
        }
        Ok(())
    }

    async fn retire_invalid_ready_workers(&self) -> Result<(), String> {
        let db = self.state.db().clone();
        let pools = tokio::task::spawn_blocking(move || db.list_fork_pools())
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        for pool in pools {
            let db = self.state.db().clone();
            let pool_name = pool.name.clone();
            let slots = tokio::task::spawn_blocking(move || db.list_fork_pool_slots(&pool_name))
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            for slot in slots
                .into_iter()
                .filter(|slot| slot.state == ForkPoolSlotState::Ready)
            {
                let db = self.state.db().clone();
                let machine = slot.machine_name.clone();
                let vm = tokio::task::spawn_blocking(move || db.get_vm(&machine))
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                let valid = vm
                    .as_ref()
                    .map(|record| {
                        record.forkpoint_held
                            && record.golden.as_deref() == Some(pool.golden.as_str())
                            && record.is_process_alive()
                    })
                    .unwrap_or(false);
                if !valid {
                    let db = self.state.db().clone();
                    let machine = slot.machine_name;
                    tokio::task::spawn_blocking(move || {
                        db.mark_fork_pool_slot_retiring(
                            &machine,
                            crate::util::current_timestamp(),
                            Some("ready worker is missing, dead, or no longer held".into()),
                        )
                    })
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }

    async fn delete_retired_workers(&self) -> Result<(), String> {
        let db = self.state.db().clone();
        let slots = tokio::task::spawn_blocking(move || db.list_retiring_fork_pool_slots())
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        for slot in slots {
            match delete_one(self.state.clone(), slot.machine_name.clone()).await {
                Ok(_) | Err(crate::api::error::ApiError::NotFound(_)) => {
                    let db = self.state.db().clone();
                    let machine = slot.machine_name;
                    tokio::task::spawn_blocking(move || db.remove_fork_pool_slot(&machine))
                        .await
                        .map_err(|e| e.to_string())?
                        .map_err(|e| e.to_string())?;
                }
                Err(error) => {
                    tracing::warn!(
                        pool = %slot.pool_name,
                        machine = %slot.machine_name,
                        error = ?error,
                        "failed to retire fork pool worker"
                    );
                }
            }
        }
        Ok(())
    }

    async fn retire_dead_leased_workers(&self) -> Result<(), String> {
        let db = self.state.db().clone();
        let leases = tokio::task::spawn_blocking(move || db.list_active_fork_leases())
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        for lease in leases {
            let db = self.state.db().clone();
            let machine = lease.machine_name.clone();
            let alive = tokio::task::spawn_blocking(move || {
                db.get_vm(&machine)
                    .map(|record| record.map(|vm| vm.is_process_alive()).unwrap_or(false))
            })
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
            if !alive {
                let db = self.state.db().clone();
                let lease_id = lease.id.clone();
                tokio::task::spawn_blocking(move || {
                    db.fail_active_fork_lease(
                        &lease_id,
                        crate::util::current_timestamp(),
                        "leased worker process exited".into(),
                    )
                })
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
                tracing::warn!(
                    pool = %lease.pool_name,
                    lease = %lease.id,
                    machine = %lease.machine_name,
                    "fork pool worker exited while leased"
                );
            }
        }
        Ok(())
    }

    async fn fill_pool(state: Arc<ApiState>, pool: ForkPoolRecord) {
        // Bound each pool's work so a large cold fill cannot starve expiry and
        // cleanup for every other pool. Successful ticks still add several
        // workers, while failures stop immediately to avoid a tight loop.
        for _ in 0..MAX_PROVISIONS_PER_POOL_TICK {
            let suffix = crate::util::generate_short_id();
            // Pool names are validated ASCII. Keep room for `pool-`, `-`, and
            // the random suffix under MAX_VM_NAME_LENGTH.
            let max_prefix = crate::data::MAX_VM_NAME_LENGTH - "pool--".len() - suffix.len();
            let prefix = &pool.name[..pool.name.len().min(max_prefix)];
            let machine = format!("pool-{prefix}-{suffix}");
            let db = state.db().clone();
            let pool_name = pool.name.clone();
            let machine_for_reservation = machine.clone();
            let reserved = match tokio::task::spawn_blocking(move || {
                db.reserve_fork_pool_slot(
                    &pool_name,
                    &machine_for_reservation,
                    crate::util::current_timestamp(),
                )
            })
            .await
            {
                Ok(Ok(reserved)) => reserved,
                Ok(Err(error)) => {
                    tracing::warn!(pool = %pool.name, %error, "failed to reserve fork pool worker");
                    break;
                }
                Err(error) => {
                    tracing::warn!(pool = %pool.name, %error, "fork pool reservation task failed");
                    break;
                }
            };
            if !reserved {
                break;
            }

            let request = ForkRequest {
                name: machine.clone(),
                ports: Vec::new(),
                share_weights: pool.share_weights,
                wait_ready: true,
                hold: true,
                ready_timeout_secs: Some(pool.ready_timeout_secs),
                env: Vec::new(),
                secrets: Default::default(),
            };
            match fork_machine_inner(state.clone(), pool.golden.clone(), request).await {
                Ok(info) if info.forkpoint_held => {
                    let db = state.db().clone();
                    let machine_ready = machine.clone();
                    match tokio::task::spawn_blocking(move || {
                        db.mark_fork_pool_slot_ready(
                            &machine_ready,
                            crate::util::current_timestamp(),
                        )
                    })
                    .await
                    {
                        Ok(Ok(true)) => {
                            tracing::info!(pool = %pool.name, machine = %machine, "fork pool worker ready");
                            // Continue immediately until the durable target says
                            // no more capacity is needed.
                            continue;
                        }
                        Ok(Ok(false)) => {
                            tracing::info!(pool = %pool.name, machine = %machine, "pool changed while worker was provisioning; retiring worker");
                        }
                        Ok(Err(error)) => {
                            tracing::warn!(pool = %pool.name, machine = %machine, %error, "failed to mark fork pool worker ready");
                        }
                        Err(error) => {
                            tracing::warn!(pool = %pool.name, machine = %machine, %error, "fork pool ready task failed");
                        }
                    }
                }
                Ok(_) => {
                    tracing::warn!(pool = %pool.name, machine = %machine, "forked pool worker was not held");
                }
                Err(error) => {
                    tracing::warn!(pool = %pool.name, machine = %machine, error = ?error, "failed to provision fork pool worker");
                    let db = state.db().clone();
                    let machine_retiring = machine.clone();
                    let message = format!("{error:?}");
                    let _ = tokio::task::spawn_blocking(move || {
                        db.mark_fork_pool_slot_retiring(
                            &machine_retiring,
                            crate::util::current_timestamp(),
                            Some(message),
                        )
                    })
                    .await;
                    // Avoid a tight failure loop when the golden is not ready.
                    break;
                }
            }

            let db = state.db().clone();
            let machine_retiring = machine;
            let _ = tokio::task::spawn_blocking(move || {
                db.mark_fork_pool_slot_retiring(
                    &machine_retiring,
                    crate::util::current_timestamp(),
                    Some("worker could not enter ready state".into()),
                )
            })
            .await;
            break;
        }
    }
}
