//! Framework-aware fused rollout executor endpoints.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use futures_util::future::join_all;
use std::sync::Arc;

use crate::api::error::ApiError;
use crate::api::rollout::{
    CreateRolloutExecutorRequest, PublishDeviceRolloutPolicyRequest, PublishRolloutPolicyRequest,
    RolloutBatchItemResponse, RolloutBatchRequest, RolloutBatchResponse, RolloutExecutorInfo,
    RolloutGenerateRequest, RolloutGenerateResponse, RolloutPolicyInfo,
};
use crate::api::state::ApiState;

const MAX_BATCH_JOBS: usize = 256;

/// Register a healthy local fused rollout backend.
#[utoipa::path(
    post,
    path = "/api/v1/rollout-executors",
    request_body = CreateRolloutExecutorRequest,
    responses(
        (status = 201, description = "Executor registered", body = RolloutExecutorInfo),
        (status = 400, description = "Invalid executor configuration"),
        (status = 409, description = "Executor already exists"),
        (status = 503, description = "Backend unavailable")
    ),
    tag = "Rollouts"
)]
pub async fn create_executor(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<CreateRolloutExecutorRequest>,
) -> Result<(StatusCode, Json<RolloutExecutorInfo>), ApiError> {
    if let Some(pool) = &request.fallback_pool {
        let db = state.db().clone();
        let pool = pool.clone();
        let exists = tokio::task::spawn_blocking(move || db.get_fork_pool(&pool))
            .await
            .map_err(ApiError::from)?
            .map_err(ApiError::database)?
            .is_some();
        if !exists {
            return Err(ApiError::BadRequest(format!(
                "fallback pool '{}' does not exist",
                request.fallback_pool.as_deref().unwrap_or_default()
            )));
        }
    }
    let info = state
        .rollout()
        .create(request)
        .await
        .map_err(ApiError::from)?;
    Ok((StatusCode::CREATED, Json(info)))
}

/// List local fused rollout executors.
#[utoipa::path(
    get,
    path = "/api/v1/rollout-executors",
    responses((status = 200, description = "Executors", body = Vec<RolloutExecutorInfo>)),
    tag = "Rollouts"
)]
pub async fn list_executors(State(state): State<Arc<ApiState>>) -> Json<Vec<RolloutExecutorInfo>> {
    Json(state.rollout().list().await)
}

/// Inspect one local fused rollout executor.
#[utoipa::path(
    get,
    path = "/api/v1/rollout-executors/{name}",
    params(("name" = String, Path, description = "Executor name")),
    responses(
        (status = 200, description = "Executor state", body = RolloutExecutorInfo),
        (status = 404, description = "Executor not found")
    ),
    tag = "Rollouts"
)]
pub async fn get_executor(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
) -> Result<Json<RolloutExecutorInfo>, ApiError> {
    let executor = state.rollout().get(&name).await.map_err(ApiError::from)?;
    Ok(Json(executor.info().await))
}

/// Delete an executor after draining requests and unloading its adapters.
#[utoipa::path(
    delete,
    path = "/api/v1/rollout-executors/{name}",
    params(("name" = String, Path, description = "Executor name")),
    responses(
        (status = 204, description = "Executor deleted"),
        (status = 404, description = "Executor not found"),
        (status = 503, description = "Adapter unload failed")
    ),
    tag = "Rollouts"
)]
pub async fn delete_executor(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, ApiError> {
    state
        .rollout()
        .delete(&name)
        .await
        .map_err(ApiError::from)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Verify, load, and atomically publish one policy adapter version.
#[utoipa::path(
    post,
    path = "/api/v1/rollout-executors/{name}/policies",
    params(("name" = String, Path, description = "Executor name")),
    request_body = PublishRolloutPolicyRequest,
    responses(
        (status = 201, description = "Policy published", body = RolloutPolicyInfo),
        (status = 400, description = "Invalid adapter or digest"),
        (status = 404, description = "Executor not found"),
        (status = 409, description = "Version conflict"),
        (status = 503, description = "Backend unavailable")
    ),
    tag = "Rollouts"
)]
pub async fn publish_policy(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Json(request): Json<PublishRolloutPolicyRequest>,
) -> Result<(StatusCode, Json<RolloutPolicyInfo>), ApiError> {
    let executor = state.rollout().get(&name).await.map_err(ApiError::from)?;
    let policy = executor
        .publish_policy(request)
        .await
        .map_err(ApiError::from)?;
    Ok((StatusCode::CREATED, Json(policy)))
}

/// Validate, redeem, and atomically publish a device-resident policy version.
#[utoipa::path(
    post,
    path = "/api/v1/rollout-executors/{name}/device-policies",
    params(("name" = String, Path, description = "Executor name")),
    request_body = PublishDeviceRolloutPolicyRequest,
    responses(
        (status = 201, description = "Device policy published", body = RolloutPolicyInfo),
        (status = 400, description = "Invalid token or tensor manifest"),
        (status = 404, description = "Executor not found"),
        (status = 409, description = "Version conflict"),
        (status = 503, description = "Device sidecar unavailable")
    ),
    tag = "Rollouts"
)]
pub async fn publish_device_policy(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Json(request): Json<PublishDeviceRolloutPolicyRequest>,
) -> Result<(StatusCode, Json<RolloutPolicyInfo>), ApiError> {
    #[cfg(target_os = "linux")]
    {
        let executor = state.rollout().get(&name).await.map_err(ApiError::from)?;
        let policy = executor
            .publish_device_policy(request)
            .await
            .map_err(ApiError::from)?;
        Ok((StatusCode::CREATED, Json(policy)))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (state, name, request);
        Err(ApiError::BadRequest(
            "device-resident rollout handoff requires Linux".into(),
        ))
    }
}

/// Stop routing one policy version and unload it after active requests drain.
#[utoipa::path(
    delete,
    path = "/api/v1/rollout-executors/{name}/policies/{policy}/{version}",
    params(
        ("name" = String, Path, description = "Executor name"),
        ("policy" = String, Path, description = "Policy name"),
        ("version" = String, Path, description = "Policy version")
    ),
    responses(
        (status = 204, description = "Policy retired"),
        (status = 404, description = "Executor or policy not found"),
        (status = 503, description = "Backend unavailable")
    ),
    tag = "Rollouts"
)]
pub async fn retire_policy(
    State(state): State<Arc<ApiState>>,
    Path((name, policy, version)): Path<(String, String, String)>,
) -> Result<StatusCode, ApiError> {
    let executor = state.rollout().get(&name).await.map_err(ApiError::from)?;
    executor
        .retire_policy(&policy, &version)
        .await
        .map_err(ApiError::from)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Generate rollouts through one explicitly versioned policy adapter.
#[utoipa::path(
    post,
    path = "/api/v1/rollout-executors/{name}/generate",
    params(("name" = String, Path, description = "Executor name")),
    request_body = RolloutGenerateRequest,
    responses(
        (status = 200, description = "Generated rollouts", body = RolloutGenerateResponse),
        (status = 400, description = "Invalid generation request"),
        (status = 404, description = "Executor or policy not found"),
        (status = 409, description = "Idempotency conflict"),
        (status = 503, description = "Queue full or backend unavailable")
    ),
    tag = "Rollouts"
)]
pub async fn generate(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Json(request): Json<RolloutGenerateRequest>,
) -> Result<Json<RolloutGenerateResponse>, ApiError> {
    let executor = state.rollout().get(&name).await.map_err(ApiError::from)?;
    let response = executor.generate(request).await.map_err(ApiError::from)?;
    Ok(Json(response))
}

/// Submit a bounded cohort concurrently so the backend can fuse policies in one batch.
#[utoipa::path(
    post,
    path = "/api/v1/rollout-executors/{name}/batches",
    params(("name" = String, Path, description = "Executor name")),
    request_body = RolloutBatchRequest,
    responses(
        (status = 200, description = "Ordered per-job results", body = RolloutBatchResponse),
        (status = 400, description = "Invalid or oversized cohort"),
        (status = 404, description = "Executor not found")
    ),
    tag = "Rollouts"
)]
pub async fn generate_batch(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Json(request): Json<RolloutBatchRequest>,
) -> Result<Json<RolloutBatchResponse>, ApiError> {
    if request.jobs.is_empty() || request.jobs.len() > MAX_BATCH_JOBS {
        return Err(ApiError::BadRequest(format!(
            "jobs must contain between 1 and {MAX_BATCH_JOBS} items"
        )));
    }
    let executor = state.rollout().get(&name).await.map_err(ApiError::from)?;
    let futures = request.jobs.into_iter().map(|job| {
        let executor = executor.clone();
        async move {
            let key = job.idempotency_key.clone();
            match executor.generate(job).await {
                Ok(response) => RolloutBatchItemResponse {
                    idempotency_key: key,
                    response: Some(response),
                    error_code: None,
                    error: None,
                },
                Err(error) => RolloutBatchItemResponse {
                    idempotency_key: key,
                    response: None,
                    error_code: Some(error.code().into()),
                    error: Some(error.message().into()),
                },
            }
        }
    });
    Ok(Json(RolloutBatchResponse {
        jobs: join_all(futures).await,
    }))
}
