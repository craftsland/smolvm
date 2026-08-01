//! Background reconciliation for automatic held-fork worker pools.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;

use crate::api::handlers::machines::{delete_one, fork_held_machines_inner};
use crate::api::state::ApiState;
use crate::pool::{ForkPoolRecord, ForkPoolSlotState};

const RECONCILE_INTERVAL: Duration = Duration::from_secs(1);
const MAX_PROVISIONS_PER_POOL_TICK: usize = 4;

/// Maintains each pool's clean-worker target and reaps finished leases.
pub struct ForkPoolController {
    state: Arc<ApiState>,
    shutdown_rx: watch::Receiver<bool>,
    fills: tokio::task::JoinSet<String>,
    filling: std::collections::HashSet<String>,
    nvml: Option<crate::api::admission::NvmlSampler>,
    host_cpu: crate::api::admission::HostCpuSampler,
}

impl ForkPoolController {
    /// Create a controller sharing the API's durable state and shutdown signal.
    pub fn new(state: Arc<ApiState>, shutdown_rx: watch::Receiver<bool>) -> Self {
        let nvml = match crate::api::admission::NvmlSampler::new() {
            Ok(nvml) => Some(nvml),
            Err(error) => {
                tracing::info!(%error, "NVML unavailable; automatic admission will use full residency");
                None
            }
        };
        Self {
            state,
            shutdown_rx,
            fills: tokio::task::JoinSet::new(),
            filling: std::collections::HashSet::new(),
            nvml,
            host_cpu: crate::api::admission::HostCpuSampler::default(),
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
        self.update_admission(&pools).await?;
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

    async fn update_admission(&mut self, pools: &[ForkPoolRecord]) -> Result<(), String> {
        let gpu = self.nvml.as_mut().and_then(|nvml| nvml.sample());
        let host_cpu = self.host_cpu.sample();
        let names = pools
            .iter()
            .map(|pool| pool.name.clone())
            .collect::<std::collections::HashSet<_>>();
        self.state.admission().retain_pools(&names);

        for pool in pools.iter().filter(|pool| !pool.deleting) {
            if !pool.auto_admission {
                self.state.admission().observe(pool, 0, 0, gpu, host_cpu);
                continue;
            }
            let db = self.state.db().clone();
            let pool_name = pool.name.clone();
            let (active, completed) =
                tokio::task::spawn_blocking(move || db.fork_pool_admission_counts(&pool_name))
                    .await
                    .map_err(|error| error.to_string())?
                    .map_err(|error| error.to_string())?;
            self.state
                .admission()
                .observe(pool, active, completed, gpu, host_cpu);
            if let Some(snapshot) = self.state.admission().snapshot(pool) {
                metrics::gauge!("smolvm_pool_admission_limit", "pool" => pool.name.clone())
                    .set(f64::from(snapshot.effective_limit));
                if let Some(utilization) = snapshot.gpu_utilization_percent {
                    metrics::gauge!("smolvm_pool_gpu_utilization_percent", "pool" => pool.name.clone())
                        .set(utilization);
                }
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
        // cleanup for every other pool. Reserve the bounded deficit first so all
        // workers in this tick can share one golden checkpoint.
        let mut machines = Vec::new();
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
            machines.push(machine);
        }
        if machines.is_empty() {
            return;
        }

        let results = match fork_held_machines_inner(
            state.clone(),
            pool.golden.clone(),
            machines.clone(),
            pool.share_weights,
            Duration::from_secs(pool.ready_timeout_secs),
        )
        .await
        {
            Ok(results) => results,
            Err(error) => {
                tracing::warn!(pool = %pool.name, error = ?error, workers = machines.len(), "failed to prepare fork pool worker batch");
                for machine in machines {
                    Self::retire_failed_provision(&state, machine, format!("{error:?}")).await;
                }
                return;
            }
        };

        for (machine, result) in results {
            let retirement_reason = match result {
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
                            continue;
                        }
                        Ok(Ok(false)) => {
                            tracing::info!(pool = %pool.name, machine = %machine, "pool changed while worker was provisioning; retiring worker");
                            "pool changed while worker was provisioning".into()
                        }
                        Ok(Err(error)) => {
                            tracing::warn!(pool = %pool.name, machine = %machine, %error, "failed to mark fork pool worker ready");
                            error.to_string()
                        }
                        Err(error) => {
                            tracing::warn!(pool = %pool.name, machine = %machine, %error, "fork pool ready task failed");
                            error.to_string()
                        }
                    }
                }
                Ok(_) => {
                    tracing::warn!(pool = %pool.name, machine = %machine, "forked pool worker was not held");
                    "forked pool worker was not held".into()
                }
                Err(error) => {
                    tracing::warn!(pool = %pool.name, machine = %machine, error = ?error, "failed to provision fork pool worker");
                    format!("{error:?}")
                }
            };
            Self::retire_failed_provision(&state, machine, retirement_reason).await;
        }
    }

    async fn retire_failed_provision(state: &Arc<ApiState>, machine: String, message: String) {
        let db = state.db().clone();
        let _ = tokio::task::spawn_blocking(move || {
            db.mark_fork_pool_slot_retiring(
                &machine,
                crate::util::current_timestamp(),
                Some(message),
            )
        })
        .await;
    }
}
