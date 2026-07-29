//! Control-initiated artifact pre-warming.
//!
//! `POST /artifacts/warm` pulls a `.smolmachine` layer blob into this node's
//! local cache *ahead of* any machine that needs it, so the create path finds a
//! warm cache instead of racing a cold multi-hundred-MB download.
//!
//! ## Why this exists
//!
//! The shared registry is reached over a link with a fixed bandwidth ceiling
//! that every concurrent pull in the fleet divides between them. A tenant who
//! pushes a fresh artifact and immediately fans out N machines makes N nodes
//! pull the same bytes at the same moment, so each one gets roughly 1/N of the
//! ceiling and the pull stretches from seconds into tens of seconds — long
//! enough to lose a client-side readiness race. Fetching once per node at push
//! time, when nothing is waiting on it, moves that transfer off the critical
//! path entirely and lets the create hit a local cache instead.
//!
//! ## Security
//!
//! Like `/drain` and `/p2p/blob/{digest}`, this route is mTLS-gated by the serve
//! listener by construction: only a client whose cert chains to the fleet
//! node-CA can reach it. The caller supplies the short-lived, tenant-scoped
//! registry token, so this node can fetch exactly the artifact it was told to
//! and nothing else — pre-warming grants no access the create path wouldn't
//! already have.

use axum::Json;

use crate::api::error::ApiError;
use crate::api::types::{WarmArtifactRequest, WarmArtifactResponse};

/// Pull `req.reference` into this node's blob cache.
///
/// Idempotent: a reference whose layer is already cached returns immediately
/// with `already_cached: true` and touches no network, so the control plane can
/// re-warm freely (on a retry, a node restart, or a repeated push) without
/// paying for the transfer twice.
pub async fn warm_artifact(
    Json(req): Json<WarmArtifactRequest>,
) -> Result<Json<WarmArtifactResponse>, ApiError> {
    if req.reference.trim().is_empty() {
        return Err(ApiError::BadRequest("reference must not be empty".into()));
    }

    tracing::info!(reference = %req.reference, "pre-warming .smolmachine artifact");

    let result = super::machines::pull_smolmachine(
        &req.reference,
        req.identity_token.as_deref(),
        &req.blob_peers,
    )
    .await?;

    tracing::info!(
        reference = %req.reference,
        digest = %result.digest,
        size_bytes = result.size,
        already_cached = result.cached,
        "artifact pre-warm complete"
    );

    Ok(Json(WarmArtifactResponse {
        digest: result.digest,
        size_bytes: result.size,
        already_cached: result.cached,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty reference is rejected at the boundary rather than handed to the
    /// reference parser. Pre-warm is fire-and-forget from the control plane's
    /// side, so a request that could never name an artifact should fail loudly
    /// here instead of turning into a confusing parse error deeper in the pull.
    #[tokio::test]
    async fn empty_reference_is_rejected() {
        for blank in ["", "   "] {
            let err = warm_artifact(Json(WarmArtifactRequest {
                reference: blank.to_string(),
                identity_token: None,
                blob_peers: Vec::new(),
            }))
            .await
            .expect_err("a blank reference must not reach the pull path");
            assert!(matches!(err, ApiError::BadRequest(_)));
        }
    }
}
