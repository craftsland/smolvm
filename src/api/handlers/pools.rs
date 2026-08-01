//! Automatic held-fork pool and one-shot lease handlers.

use axum::{
    extract::{Path, Query, State},
    Json,
};
use std::sync::Arc;

use crate::api::error::ApiError;
use crate::api::state::ApiState;
use crate::api::types::{
    AcquireForkLeaseRequest, ApiErrorResponse, CreateForkPoolRequest, DeleteForkPoolQuery,
    DeleteResponse, ForkLeaseInfo, ForkPoolInfo, ListForkPoolsResponse, ResizeForkPoolRequest,
};
use crate::data::validate_vm_name;
use crate::pool::{
    ClaimForkPoolSlot, ForkLeaseRecord, ForkLeaseState, ForkPoolRecord, ForkPoolSlotState,
};

const DEFAULT_READY_TIMEOUT_SECS: u64 = 240;
const MAX_READY_TIMEOUT_SECS: u64 = 60 * 60;
const DEFAULT_LEASE_TTL_SECS: u64 = 300;
const MAX_POOL_READY: u32 = 256;
const MIN_LEASE_TTL_SECS: u64 = 30;
const MAX_LEASE_TTL_SECS: u64 = 24 * 60 * 60;
const MAX_IDEMPOTENCY_KEY_BYTES: usize = 256;

fn lease_info(lease: ForkLeaseRecord) -> ForkLeaseInfo {
    ForkLeaseInfo {
        id: lease.id,
        pool: lease.pool_name,
        machine: lease.machine_name,
        state: lease.state.as_str().to_string(),
        created_at: lease.created_at,
        expires_at: lease.expires_at,
        error: lease.last_error,
    }
}

async fn activate_claimed_lease(
    state: Arc<ApiState>,
    lease: ForkLeaseRecord,
    assignment: Vec<(String, String)>,
) -> Result<ForkLeaseRecord, String> {
    let record = match state.lookup_vm(&lease.machine_name).await {
        Ok(Some(record)) => record,
        Ok(None) => {
            let message = "claimed pool worker disappeared".to_string();
            let db = state.db().clone();
            let lease_id = lease.id.clone();
            let persisted = message.clone();
            tokio::task::spawn_blocking(move || {
                db.fail_fork_lease(&lease_id, crate::util::current_timestamp(), persisted)
            })
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
            return Err(message);
        }
        Err(error) => return Err(format!("pool worker lookup failed: {error:?}")),
    };
    let machine = lease.machine_name.clone();
    let activation = tokio::task::spawn_blocking(move || {
        crate::agent::fork::activate_held_fork(&machine, &record, &assignment)
    })
    .await
    .map_err(|e| format!("pool activation task failed: {e}"))?;
    if let Err(error) = activation {
        let message = error.to_string();
        let db = state.db().clone();
        let lease_id = lease.id.clone();
        let persisted = message.clone();
        tokio::task::spawn_blocking(move || {
            db.fail_fork_lease(&lease_id, crate::util::current_timestamp(), persisted)
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
        return Err(format!(
            "pool worker was consumed and will be replaced after activation failed: {message}"
        ));
    }
    let db = state.db().clone();
    let lease_id = lease.id.clone();
    let active = tokio::task::spawn_blocking(move || {
        db.mark_fork_lease_active(&lease_id, crate::util::current_timestamp())
    })
    .await
    .map_err(|e| format!("lease activation commit task failed: {e}"))?
    .map_err(|e| e.to_string())?
    .ok_or_else(|| "claimed lease disappeared".to_string())?;
    if active.state != ForkLeaseState::Active {
        return Err(format!(
            "lease changed to '{}' before activation completed",
            active.state.as_str()
        ));
    }
    Ok(active)
}

async fn pool_info(state: &ApiState, pool: ForkPoolRecord) -> Result<ForkPoolInfo, ApiError> {
    let db = state.db().clone();
    let pool_name = pool.name.clone();
    let slots = tokio::task::spawn_blocking(move || db.list_fork_pool_slots(&pool_name))
        .await
        .map_err(|e| ApiError::internal(format!("pool slot query task failed: {e}")))?
        .map_err(ApiError::database)?;
    let mut provisioning = 0;
    let mut ready = 0;
    let mut activating = 0;
    let mut active = 0;
    let mut retiring = 0;
    for slot in slots {
        match slot.state {
            ForkPoolSlotState::Provisioning => provisioning += 1,
            ForkPoolSlotState::Ready => ready += 1,
            ForkPoolSlotState::Activating => activating += 1,
            ForkPoolSlotState::Leased => active += 1,
            ForkPoolSlotState::Retiring => retiring += 1,
        }
    }
    Ok(ForkPoolInfo {
        name: pool.name,
        golden: pool.golden,
        desired_ready: pool.desired_ready,
        max_active: pool.max_active,
        share_weights: pool.share_weights,
        lease_ttl_secs: pool.lease_ttl_secs,
        provisioning,
        ready,
        activating,
        active,
        retiring,
        deleting: pool.deleting,
        created_at: pool.created_at,
    })
}

fn validate_ttl(ttl: u64) -> Result<u64, ApiError> {
    if !(MIN_LEASE_TTL_SECS..=MAX_LEASE_TTL_SECS).contains(&ttl) {
        return Err(ApiError::BadRequest(format!(
            "lease TTL must be between {MIN_LEASE_TTL_SECS} and {MAX_LEASE_TTL_SECS} seconds"
        )));
    }
    Ok(ttl)
}

/// Create an automatically replenished held-fork pool.
#[utoipa::path(
    post,
    path = "/api/v1/pools",
    tag = "Pools",
    request_body = CreateForkPoolRequest,
    responses(
        (status = 200, description = "Pool accepted for asynchronous fill", body = ForkPoolInfo),
        (status = 400, description = "Invalid pool configuration", body = ApiErrorResponse),
        (status = 404, description = "Golden machine not found", body = ApiErrorResponse),
        (status = 409, description = "Pool already exists or golden is invalid", body = ApiErrorResponse)
    )
)]
pub async fn create_pool(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<CreateForkPoolRequest>,
) -> Result<Json<ForkPoolInfo>, ApiError> {
    validate_vm_name(&req.name, "pool name").map_err(ApiError::BadRequest)?;
    validate_vm_name(&req.golden, "golden machine name").map_err(ApiError::BadRequest)?;
    if req.desired_ready == 0 || req.desired_ready > MAX_POOL_READY {
        return Err(ApiError::BadRequest(format!(
            "desiredReady must be between 1 and {MAX_POOL_READY}"
        )));
    }
    if matches!(req.max_active, Some(0)) {
        return Err(ApiError::BadRequest(
            "maxActive must be greater than zero when set".into(),
        ));
    }
    let ready_timeout_secs = req.ready_timeout_secs.unwrap_or(DEFAULT_READY_TIMEOUT_SECS);
    if ready_timeout_secs == 0 || ready_timeout_secs > MAX_READY_TIMEOUT_SECS {
        return Err(ApiError::BadRequest(format!(
            "readyTimeoutSecs must be between 1 and {MAX_READY_TIMEOUT_SECS}"
        )));
    }
    let lease_ttl_secs = validate_ttl(req.lease_ttl_secs.unwrap_or(DEFAULT_LEASE_TTL_SECS))?;
    let golden = state
        .lookup_vm(&req.golden)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("machine '{}' not found", req.golden)))?;
    if golden.golden.is_some() {
        return Err(ApiError::Conflict(
            "a fork clone cannot be used as a pool golden".into(),
        ));
    }
    if !golden.is_process_alive() {
        return Err(ApiError::Conflict(format!(
            "golden machine '{}' is not running",
            req.golden
        )));
    }
    if req.share_weights && !golden.cuda {
        return Err(ApiError::Conflict(format!(
            "golden machine '{}' does not have CUDA enabled",
            req.golden
        )));
    }
    let golden_name = req.golden.clone();
    let forkable = tokio::task::spawn_blocking(move || {
        let control = crate::agent::fork::control_socket_path(&golden_name);
        if !control.exists() {
            return false;
        }
        crate::agent::fork::control_socket_cmd(&control, "STATUS")
            .map(|status| status.starts_with("OK"))
            .unwrap_or(false)
    })
    .await
    .map_err(|e| ApiError::internal(format!("golden forkability task failed: {e}")))?;
    if !forkable {
        return Err(ApiError::Conflict(format!(
            "golden machine '{}' is not running forkable",
            req.golden
        )));
    }
    let pool = ForkPoolRecord {
        name: req.name,
        golden: req.golden,
        desired_ready: req.desired_ready,
        max_active: req.max_active,
        share_weights: req.share_weights,
        ready_timeout_secs,
        lease_ttl_secs,
        created_at: crate::util::current_timestamp(),
        deleting: false,
    };
    let db = state.db().clone();
    let inserted_pool = pool.clone();
    let inserted =
        tokio::task::spawn_blocking(move || db.insert_fork_pool_if_not_exists(&inserted_pool))
            .await
            .map_err(|e| ApiError::internal(format!("pool insert task failed: {e}")))?
            .map_err(ApiError::database)?;
    if !inserted {
        return Err(ApiError::Conflict(format!(
            "fork pool '{}' already exists",
            pool.name
        )));
    }
    Ok(Json(pool_info(&state, pool).await?))
}

/// List automatic fork pools.
#[utoipa::path(
    get,
    path = "/api/v1/pools",
    tag = "Pools",
    responses((status = 200, description = "Pool list", body = ListForkPoolsResponse))
)]
pub async fn list_pools(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ListForkPoolsResponse>, ApiError> {
    let db = state.db().clone();
    let pools = tokio::task::spawn_blocking(move || db.list_fork_pools())
        .await
        .map_err(|e| ApiError::internal(format!("pool list task failed: {e}")))?
        .map_err(ApiError::database)?;
    let mut infos = Vec::with_capacity(pools.len());
    for pool in pools {
        infos.push(pool_info(&state, pool).await?);
    }
    Ok(Json(ListForkPoolsResponse { pools: infos }))
}

/// Get one automatic fork pool.
#[utoipa::path(
    get,
    path = "/api/v1/pools/{name}",
    tag = "Pools",
    params(("name" = String, Path, description = "Pool name")),
    responses(
        (status = 200, description = "Pool state", body = ForkPoolInfo),
        (status = 404, description = "Pool not found", body = ApiErrorResponse)
    )
)]
pub async fn get_pool(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
) -> Result<Json<ForkPoolInfo>, ApiError> {
    let db = state.db().clone();
    let lookup = name.clone();
    let pool = tokio::task::spawn_blocking(move || db.get_fork_pool(&lookup))
        .await
        .map_err(|e| ApiError::internal(format!("pool lookup task failed: {e}")))?
        .map_err(ApiError::database)?
        .ok_or_else(|| ApiError::NotFound(format!("fork pool '{name}' not found")))?;
    Ok(Json(pool_info(&state, pool).await?))
}

/// Change a pool's clean-worker target.
#[utoipa::path(
    put,
    path = "/api/v1/pools/{name}/size",
    tag = "Pools",
    params(("name" = String, Path, description = "Pool name")),
    request_body = ResizeForkPoolRequest,
    responses(
        (status = 200, description = "Updated pool state", body = ForkPoolInfo),
        (status = 400, description = "Invalid target", body = ApiErrorResponse),
        (status = 404, description = "Pool not found", body = ApiErrorResponse),
        (status = 409, description = "Pool is deleting", body = ApiErrorResponse)
    )
)]
pub async fn resize_pool(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Json(req): Json<ResizeForkPoolRequest>,
) -> Result<Json<ForkPoolInfo>, ApiError> {
    if req.desired_ready > MAX_POOL_READY {
        return Err(ApiError::BadRequest(format!(
            "desiredReady must be at most {MAX_POOL_READY}"
        )));
    }
    let db = state.db().clone();
    let pool_name = name.clone();
    let pool = tokio::task::spawn_blocking(move || {
        db.resize_fork_pool(
            &pool_name,
            req.desired_ready,
            crate::util::current_timestamp(),
        )
    })
    .await
    .map_err(|e| ApiError::internal(format!("pool resize task failed: {e}")))?
    .map_err(ApiError::database)?
    .ok_or_else(|| ApiError::NotFound(format!("fork pool '{name}' not found")))?;
    if pool.deleting {
        return Err(ApiError::Conflict(format!(
            "fork pool '{name}' is deleting"
        )));
    }
    Ok(Json(pool_info(&state, pool).await?))
}

/// Begin asynchronous pool deletion.
#[utoipa::path(
    delete,
    path = "/api/v1/pools/{name}",
    tag = "Pools",
    params(
        ("name" = String, Path, description = "Pool name"),
        ("force" = Option<bool>, Query, description = "Cancel active leases")
    ),
    responses(
        (status = 200, description = "Pool deletion started", body = DeleteResponse),
        (status = 404, description = "Pool not found", body = ApiErrorResponse),
        (status = 409, description = "Pool has active leases", body = ApiErrorResponse)
    )
)]
pub async fn delete_pool(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Query(query): Query<DeleteForkPoolQuery>,
) -> Result<Json<DeleteResponse>, ApiError> {
    let db = state.db().clone();
    let pool_name = name.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        db.begin_delete_fork_pool(&pool_name, query.force, crate::util::current_timestamp())
    })
    .await
    .map_err(|e| ApiError::internal(format!("pool deletion task failed: {e}")))?
    .map_err(ApiError::database)?;
    match outcome {
        None => Err(ApiError::NotFound(format!("fork pool '{name}' not found"))),
        Some(false) => Err(ApiError::Conflict(format!(
            "fork pool '{name}' has active leases; complete them or use force=true"
        ))),
        Some(true) => Ok(Json(DeleteResponse { deleted: name })),
    }
}

/// Acquire and release one clean worker exactly once.
#[utoipa::path(
    post,
    path = "/api/v1/pools/{name}/leases",
    tag = "Pools",
    params(("name" = String, Path, description = "Pool name")),
    request_body = AcquireForkLeaseRequest,
    responses(
        (status = 200, description = "Worker lease", body = ForkLeaseInfo),
        (status = 400, description = "Invalid assignment", body = ApiErrorResponse),
        (status = 404, description = "Pool not found", body = ApiErrorResponse),
        (status = 409, description = "Pool at active-lease capacity", body = ApiErrorResponse),
        (status = 503, description = "No clean worker ready yet", body = ApiErrorResponse)
    )
)]
pub async fn acquire_lease(
    State(state): State<Arc<ApiState>>,
    Path(pool_name): Path<String>,
    Json(req): Json<AcquireForkLeaseRequest>,
) -> Result<Json<ForkLeaseInfo>, ApiError> {
    if req.idempotency_key.is_empty()
        || req.idempotency_key.len() > MAX_IDEMPOTENCY_KEY_BYTES
        || req.idempotency_key.chars().any(char::is_control)
    {
        return Err(ApiError::BadRequest(format!(
            "idempotencyKey must contain 1-{MAX_IDEMPOTENCY_KEY_BYTES} non-control bytes"
        )));
    }
    let assignment = crate::util::parse_env_list(&req.env);
    crate::agent::fork::validate_fork_env(&assignment)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let db = state.db().clone();
    let lookup = pool_name.clone();
    let pool = tokio::task::spawn_blocking(move || db.get_fork_pool(&lookup))
        .await
        .map_err(|e| ApiError::internal(format!("pool lookup task failed: {e}")))?
        .map_err(ApiError::database)?
        .ok_or_else(|| ApiError::NotFound(format!("fork pool '{pool_name}' not found")))?;
    let ttl = validate_ttl(req.ttl_secs.unwrap_or(pool.lease_ttl_secs))?;
    let lease_id = format!(
        "lease-{}{}",
        crate::util::generate_short_id(),
        crate::util::generate_short_id()
    );
    let now = crate::util::current_timestamp();
    let db = state.db().clone();
    let pool_for_claim = pool_name.clone();
    let key = req.idempotency_key.clone();
    let assignment_for_claim = assignment.clone();
    let claim = tokio::task::spawn_blocking(move || {
        db.claim_fork_pool_slot(
            &pool_for_claim,
            &lease_id,
            &key,
            &assignment_for_claim,
            ttl,
            now,
        )
    })
    .await
    .map_err(|e| ApiError::internal(format!("pool claim task failed: {e}")))?
    .map_err(ApiError::database)?;
    let lease = match claim {
        ClaimForkPoolSlot::Existing(lease) => {
            if lease.assignment != assignment || lease.ttl_secs != ttl {
                return Err(ApiError::Conflict(
                    "idempotencyKey was already used with a different assignment or TTL".into(),
                ));
            }
            return Ok(Json(lease_info(lease)));
        }
        ClaimForkPoolSlot::NoReadySlot => {
            return Err(ApiError::Unavailable(format!(
                "fork pool '{pool_name}' has no clean worker ready"
            )))
        }
        ClaimForkPoolSlot::AtCapacity => {
            return Err(ApiError::Conflict(format!(
                "fork pool '{pool_name}' reached maxActive"
            )))
        }
        ClaimForkPoolSlot::PoolNotFound => {
            return Err(ApiError::NotFound(format!(
                "fork pool '{pool_name}' not found"
            )))
        }
        ClaimForkPoolSlot::PoolDeleting => {
            return Err(ApiError::Conflict(format!(
                "fork pool '{pool_name}' is deleting"
            )))
        }
        ClaimForkPoolSlot::Claimed(lease) => lease,
    };

    // Reflect the durable claim in the in-memory fast path before publishing
    // the guest release marker. The authoritative held bit is already false in
    // SQLite, so a restart cannot resurrect this worker as ready.
    if let Ok(entry) = state.get_machine(&lease.machine_name) {
        entry.lock().forkpoint_held = false;
    }
    // Run activation in its own task. Dropping an HTTP request future does not
    // cancel this task, so a client disconnect after the durable claim cannot
    // strand a successfully released guest forever in `activating` state.
    let active = tokio::spawn(activate_claimed_lease(state.clone(), lease, assignment))
        .await
        .map_err(|e| ApiError::internal(format!("pool activation task failed: {e}")))?
        .map_err(ApiError::Internal)?;
    Ok(Json(lease_info(active)))
}

/// Get one lease's durable state.
#[utoipa::path(
    get,
    path = "/api/v1/pools/{name}/leases/{lease}",
    tag = "Pools",
    params(
        ("name" = String, Path, description = "Pool name"),
        ("lease" = String, Path, description = "Lease ID")
    ),
    responses(
        (status = 200, description = "Lease state", body = ForkLeaseInfo),
        (status = 404, description = "Lease not found", body = ApiErrorResponse)
    )
)]
pub async fn get_lease(
    State(state): State<Arc<ApiState>>,
    Path((pool_name, lease_id)): Path<(String, String)>,
) -> Result<Json<ForkLeaseInfo>, ApiError> {
    let db = state.db().clone();
    let lookup_pool = pool_name.clone();
    let lookup_lease = lease_id.clone();
    let lease = tokio::task::spawn_blocking(move || db.get_fork_lease(&lookup_pool, &lookup_lease))
        .await
        .map_err(|e| ApiError::internal(format!("lease lookup task failed: {e}")))?
        .map_err(ApiError::database)?
        .ok_or_else(|| {
            ApiError::NotFound(format!(
                "lease '{lease_id}' not found in fork pool '{pool_name}'"
            ))
        })?;
    Ok(Json(lease_info(lease)))
}

/// Extend one active lease's expiry.
#[utoipa::path(
    post,
    path = "/api/v1/pools/{name}/leases/{lease}/heartbeat",
    tag = "Pools",
    params(
        ("name" = String, Path, description = "Pool name"),
        ("lease" = String, Path, description = "Lease ID")
    ),
    responses(
        (status = 200, description = "Extended lease", body = ForkLeaseInfo),
        (status = 404, description = "Lease not found", body = ApiErrorResponse),
        (status = 409, description = "Lease is no longer active", body = ApiErrorResponse)
    )
)]
pub async fn heartbeat_lease(
    State(state): State<Arc<ApiState>>,
    Path((pool_name, lease_id)): Path<(String, String)>,
) -> Result<Json<ForkLeaseInfo>, ApiError> {
    let now = crate::util::current_timestamp();
    let db = state.db().clone();
    let lookup_pool = pool_name.clone();
    let lookup_lease = lease_id.clone();
    let lease = tokio::task::spawn_blocking(move || {
        db.heartbeat_fork_lease(&lookup_pool, &lookup_lease, now)
    })
    .await
    .map_err(|e| ApiError::internal(format!("lease heartbeat task failed: {e}")))?
    .map_err(ApiError::database)?
    .ok_or_else(|| ApiError::NotFound(format!("fork lease '{lease_id}' not found")))?;
    if lease.state != ForkLeaseState::Active || lease.expires_at <= now {
        return Err(ApiError::Conflict(format!(
            "fork lease '{lease_id}' is no longer active"
        )));
    }
    let worker_alive = state
        .lookup_vm(&lease.machine_name)
        .await?
        .map(|record| record.is_process_alive())
        .unwrap_or(false);
    if !worker_alive {
        let db = state.db().clone();
        let failed_lease = lease.id.clone();
        tokio::task::spawn_blocking(move || {
            db.fail_active_fork_lease(
                &failed_lease,
                crate::util::current_timestamp(),
                "leased worker process exited".into(),
            )
        })
        .await
        .map_err(|e| ApiError::internal(format!("failed lease task failed: {e}")))?
        .map_err(ApiError::database)?;
        return Err(ApiError::Conflict(format!(
            "fork lease '{lease_id}' worker is no longer running"
        )));
    }
    Ok(Json(lease_info(lease)))
}

/// Complete one active lease and asynchronously replace its worker.
#[utoipa::path(
    post,
    path = "/api/v1/pools/{name}/leases/{lease}/complete",
    tag = "Pools",
    params(
        ("name" = String, Path, description = "Pool name"),
        ("lease" = String, Path, description = "Lease ID")
    ),
    responses(
        (status = 200, description = "Completed lease", body = ForkLeaseInfo),
        (status = 404, description = "Lease not found", body = ApiErrorResponse),
        (status = 409, description = "Lease is not active", body = ApiErrorResponse)
    )
)]
pub async fn complete_lease(
    State(state): State<Arc<ApiState>>,
    Path((pool_name, lease_id)): Path<(String, String)>,
) -> Result<Json<ForkLeaseInfo>, ApiError> {
    let db = state.db().clone();
    let complete_pool = pool_name.clone();
    let complete_lease = lease_id.clone();
    let lease = tokio::task::spawn_blocking(move || {
        db.complete_fork_lease(
            &complete_pool,
            &complete_lease,
            crate::util::current_timestamp(),
        )
    })
    .await
    .map_err(|e| ApiError::internal(format!("lease completion task failed: {e}")))?
    .map_err(ApiError::database)?
    .ok_or_else(|| ApiError::NotFound(format!("fork lease '{lease_id}' not found")))?;
    if lease.state != ForkLeaseState::Completed {
        return Err(ApiError::Conflict(format!(
            "fork lease '{lease_id}' is '{}', not active",
            lease.state.as_str()
        )));
    }
    Ok(Json(lease_info(lease)))
}
