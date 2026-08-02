//! Durable records for automatic held-fork pools and their one-shot leases.

use serde::{Deserialize, Serialize};

/// Configuration and lifecycle state for one automatic fork pool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForkPoolRecord {
    /// Stable API name for the pool.
    pub name: String,
    /// Forkable machine used to replenish clean slots.
    pub golden: String,
    /// Number of clean, held workers the controller keeps ready.
    pub desired_ready: u32,
    /// Optional limit on simultaneously active leases.
    pub max_active: Option<u32>,
    /// Dynamically calibrate the active-lease limit from host/GPU telemetry.
    /// Absent on records written before automatic admission was introduced.
    #[serde(default)]
    pub auto_admission: bool,
    /// Share the golden's immutable CUDA allocations with each worker.
    pub share_weights: bool,
    /// Maximum time to wait for the golden's workload forkpoint.
    pub ready_timeout_secs: u64,
    /// Default time an acquired worker may run without a heartbeat.
    pub lease_ttl_secs: u64,
    /// Unix timestamp when the pool was created.
    pub created_at: u64,
    /// True after deletion has begun; no new workers or leases are admitted.
    pub deleting: bool,
}

/// Controller lifecycle state for one machine owned by a fork pool.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForkPoolSlotState {
    /// A machine name is reserved and the held fork is being created.
    Provisioning,
    /// The machine is booted and parked at its inherited forkpoint.
    Ready,
    /// The durable claim committed and guest activation is in progress.
    Activating,
    /// The worker was released to exactly one lease.
    Leased,
    /// The worker must be deleted and replaced, never reused.
    Retiring,
}

impl ForkPoolSlotState {
    /// Stable database representation used for indexed state queries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Provisioning => "provisioning",
            Self::Ready => "ready",
            Self::Activating => "activating",
            Self::Leased => "leased",
            Self::Retiring => "retiring",
        }
    }
}

/// Durable ownership record for one pool-managed machine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForkPoolSlotRecord {
    /// Pool that owns the machine.
    pub pool_name: String,
    /// Machine name in the normal smolvm machine registry.
    pub machine_name: String,
    /// Current controller lifecycle state.
    pub state: ForkPoolSlotState,
    /// Lease currently owning the machine, if it has been claimed.
    pub lease_id: Option<String>,
    /// Unix timestamp when the slot reservation was created.
    pub created_at: u64,
    /// Unix timestamp of the last state transition.
    pub updated_at: u64,
    /// Last provisioning or activation error, if any.
    pub last_error: Option<String>,
}

/// Lifecycle state for a one-shot worker lease.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForkLeaseState {
    /// The durable claim committed and guest activation is in progress.
    Activating,
    /// The guest was released and the caller owns the worker.
    Active,
    /// The caller completed the lease normally.
    Completed,
    /// The caller stopped heartbeating before the deadline.
    Expired,
    /// Activation failed after the slot was consumed.
    Failed,
    /// Pool deletion revoked the lease.
    Cancelled,
}

impl ForkLeaseState {
    /// Stable database representation used for indexed state queries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Activating => "activating",
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Expired => "expired",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Durable exactly-once claim for one pool worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForkLeaseRecord {
    /// Opaque lease identifier.
    pub id: String,
    /// Pool the worker came from.
    pub pool_name: String,
    /// Claimed worker machine name.
    pub machine_name: String,
    /// Caller-provided retry key, unique within the pool.
    pub idempotency_key: String,
    /// Current lease lifecycle state.
    pub state: ForkLeaseState,
    /// Canonical assignment environment written before guest release.
    pub assignment: Vec<(String, String)>,
    /// Digest of the canonical pre-release file payload, for retry validation.
    #[serde(default)]
    pub payload_sha256: Option<String>,
    /// Unix timestamp when the claim was created.
    pub created_at: u64,
    /// Unix timestamp of the last state transition or heartbeat.
    pub updated_at: u64,
    /// Unix timestamp after which the controller may retire the worker.
    pub expires_at: u64,
    /// Lease duration applied by each heartbeat.
    pub ttl_secs: u64,
    /// Activation failure, when the lease ended in `failed`.
    pub last_error: Option<String>,
}

/// Result of an atomic lease acquisition attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimForkPoolSlot {
    /// A new slot was durably consumed for this request.
    Claimed(ForkLeaseRecord),
    /// The same idempotency key already owns this lease.
    Existing(ForkLeaseRecord),
    /// No clean held slot is currently ready.
    NoReadySlot,
    /// The pool reached its configured active-lease limit.
    AtCapacity,
    /// The pool was removed between request validation and claim.
    PoolNotFound,
    /// Pool deletion began before the claim committed.
    PoolDeleting,
    /// Payload staging cannot safely target an externally mounted workspace.
    WorkspaceExternallyMounted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_lease_without_payload_digest_still_deserializes() {
        let lease: ForkLeaseRecord = serde_json::from_str(
            r#"{
                "id":"lease-1",
                "pool_name":"pool",
                "machine_name":"worker",
                "idempotency_key":"request",
                "state":"active",
                "assignment":[],
                "created_at":1,
                "updated_at":1,
                "expires_at":61,
                "ttl_secs":60,
                "last_error":null
            }"#,
        )
        .unwrap();
        assert_eq!(lease.payload_sha256, None);
    }

    #[test]
    fn legacy_pool_without_auto_admission_stays_full_residency() {
        let pool: ForkPoolRecord = serde_json::from_str(
            r#"{
                "name":"rollouts",
                "golden":"golden",
                "desired_ready":8,
                "max_active":null,
                "share_weights":true,
                "ready_timeout_secs":240,
                "lease_ttl_secs":300,
                "created_at":1,
                "deleting":false
            }"#,
        )
        .unwrap();
        assert!(!pool.auto_admission);
    }
}
