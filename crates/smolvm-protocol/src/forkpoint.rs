//! Stable guest paths used to coordinate a live fork.

/// Directory privately inherited by each restored VM.
pub const STATE_DIR: &str = "/run/smolvm/forkpoint";

/// Marker written by the workload when it reaches a safe fork boundary.
pub const READY_PATH: &str = "/run/smolvm/forkpoint/ready";

/// Marker written after a restored clone can safely enter ordinary timed waits.
pub const RESTORED_PATH: &str = "/run/smolvm/forkpoint/restored";

/// Marker written by the host after a clone is ready to resume.
pub const RELEASE_PATH: &str = "/run/smolvm/forkpoint/release";

/// Workload-facing helper installed in bare VMs and workload containers.
pub const HELPER_PATH: &str = "/usr/local/bin/smolvm-fork-ready";
