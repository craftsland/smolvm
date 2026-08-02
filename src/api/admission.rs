//! Lease-aware admission for CUDA fork pools.
//!
//! The controller deliberately gates durable lease claims, not CUDA daemon
//! connections.  A claimed worker must always be allowed to finish activation;
//! moving the gate below that boundary can deadlock the daemon accept loop and
//! leaves the lease state ambiguous after a restart.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::pool::ForkPoolRecord;

const OBSERVATION_WINDOW: Duration = Duration::from_secs(8);
const MIN_STABLE_SAMPLES: u32 = 5;
const PRESSURE_TTL: Duration = Duration::from_secs(30);
const REPROBE_INTERVAL: Duration = Duration::from_secs(15 * 60);
#[cfg(any(target_os = "linux", test))]
const MIN_VRAM_RESERVE_MIB: u64 = 8 * 1024;
#[cfg(any(target_os = "linux", test))]
const VRAM_RESERVE_PERCENT: u64 = 10;
const CPU_SATURATION_PERCENT: f64 = 90.0;
const MARGINAL_GAIN_PERCENT: f64 = 2.0;

/// One node-wide GPU sample. Utilization and public memory counters remain
/// aggregate, while admission uses the least-safe individual device because
/// the current runtime does not yet have a durable pool-to-device mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSample {
    /// Memory-weighted mean streaming-multiprocessor utilization.
    pub utilization_percent: f64,
    /// Aggregate used device memory in MiB.
    pub used_memory_mib: u64,
    /// Aggregate device-memory capacity in MiB.
    pub total_memory_mib: u64,
    /// Free memory on the device closest to violating its own reserve.
    limiting_device_free_memory_mib: u64,
    /// Capacity-derived reserve required on that limiting device.
    limiting_device_reserve_mib: u64,
}

impl GpuSample {
    #[cfg(any(target_os = "linux", test))]
    fn from_devices(devices: impl IntoIterator<Item = GpuDeviceSample>) -> Option<Self> {
        let mut count = 0_u32;
        let mut total_memory_mib = 0_u64;
        let mut used_memory_mib = 0_u64;
        let mut weighted_utilization = 0_f64;
        let mut limiting_device = None::<(i128, u64, u64)>;

        for device in devices {
            count = count.saturating_add(1);
            total_memory_mib = total_memory_mib.saturating_add(device.total_memory_mib);
            used_memory_mib = used_memory_mib.saturating_add(device.used_memory_mib);
            weighted_utilization += device.utilization_percent * device.total_memory_mib as f64;

            let reserve_mib = device_reserve_mib(device.total_memory_mib);
            let headroom = i128::from(device.free_memory_mib) - i128::from(reserve_mib);
            if limiting_device.is_none_or(|(least_headroom, _, _)| headroom < least_headroom) {
                limiting_device = Some((headroom, device.free_memory_mib, reserve_mib));
            }
        }

        let (_, limiting_device_free_memory_mib, limiting_device_reserve_mib) = limiting_device?;
        (count > 0 && total_memory_mib > 0).then_some(Self {
            utilization_percent: weighted_utilization / total_memory_mib as f64,
            used_memory_mib,
            total_memory_mib,
            limiting_device_free_memory_mib,
            limiting_device_reserve_mib,
        })
    }

    fn memory_safe(self) -> bool {
        self.limiting_device_free_memory_mib >= self.limiting_device_reserve_mib
    }
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, Copy)]
struct GpuDeviceSample {
    utilization_percent: f64,
    used_memory_mib: u64,
    free_memory_mib: u64,
    total_memory_mib: u64,
}

#[cfg(any(target_os = "linux", test))]
fn device_reserve_mib(total_memory_mib: u64) -> u64 {
    MIN_VRAM_RESERVE_MIB.max(total_memory_mib.saturating_mul(VRAM_RESERVE_PERCENT) / 100)
}

/// Public, read-only controller state returned with pool status.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionSnapshot {
    /// Maximum active or activating leases currently admitted.
    pub effective_limit: u32,
    /// Explanation of the most recent controller decision.
    pub reason: String,
    /// Whether the controller is comparing a candidate resident level.
    pub calibrating: bool,
    /// Latest aggregate GPU utilization, when NVML is healthy.
    pub gpu_utilization_percent: Option<f64>,
    /// Latest aggregate used GPU memory in MiB.
    pub gpu_memory_used_mib: Option<u64>,
    /// Latest aggregate GPU memory capacity in MiB.
    pub gpu_memory_total_mib: Option<u64>,
    /// Latest host CPU busy percentage.
    pub host_cpu_percent: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct Score {
    gpu_utilization_percent: f64,
    completion_rate: Option<f64>,
}

impl Score {
    fn improvement_over(self, baseline: Self) -> f64 {
        let (current, previous) = match (self.completion_rate, baseline.completion_rate) {
            (Some(current), Some(previous)) if previous > 0.0 => (current, previous),
            _ if baseline.gpu_utilization_percent > 0.0 => (
                self.gpu_utilization_percent,
                baseline.gpu_utilization_percent,
            ),
            _ => return f64::INFINITY,
        };
        ((current - previous) / previous) * 100.0
    }
}

#[derive(Debug)]
struct PoolAdmissionState {
    ceiling: u32,
    effective_limit: u32,
    best_limit: u32,
    best_score: Option<Score>,
    testing_from: Option<(u32, Score)>,
    last_lower: Option<(u32, Score)>,
    settled: bool,
    last_probe: Instant,
    last_blocked: Option<Instant>,
    observed_active: u32,
    window_started: Instant,
    window_samples: u32,
    gpu_utilization_sum: f64,
    host_cpu_sum: f64,
    completed_at_start: u64,
    latest_gpu: Option<GpuSample>,
    latest_host_cpu_percent: Option<f64>,
    telemetry_failed_open: bool,
    reason: String,
}

impl PoolAdmissionState {
    fn initial_limit(ceiling: u32) -> u32 {
        // Calibrate in at most three multiplicative steps without embedding a
        // GPU- or workload-specific resident count. Small pools stay unmodified.
        if ceiling <= 2 {
            ceiling
        } else {
            ceiling.div_ceil(3).max(1)
        }
    }

    fn new(ceiling: u32, now: Instant, completed_total: u64) -> Self {
        let effective_limit = Self::initial_limit(ceiling);
        Self {
            ceiling,
            effective_limit,
            best_limit: effective_limit,
            best_score: None,
            testing_from: None,
            last_lower: None,
            settled: effective_limit == ceiling,
            last_probe: now,
            last_blocked: None,
            observed_active: 0,
            window_started: now,
            window_samples: 0,
            gpu_utilization_sum: 0.0,
            host_cpu_sum: 0.0,
            completed_at_start: completed_total,
            latest_gpu: None,
            latest_host_cpu_percent: None,
            telemetry_failed_open: false,
            reason: "waiting for GPU telemetry".into(),
        }
    }

    fn reset_window(&mut self, now: Instant, active: u32, completed_total: u64) {
        self.observed_active = active;
        self.window_started = now;
        self.window_samples = 0;
        self.gpu_utilization_sum = 0.0;
        self.host_cpu_sum = 0.0;
        self.completed_at_start = completed_total;
    }

    fn has_pressure(&self, now: Instant) -> bool {
        self.last_blocked
            .is_some_and(|blocked| now.saturating_duration_since(blocked) <= PRESSURE_TTL)
    }

    fn update_ceiling(&mut self, ceiling: u32, now: Instant, completed_total: u64) {
        if self.ceiling == ceiling {
            return;
        }
        self.ceiling = ceiling;
        self.effective_limit = self.effective_limit.min(ceiling).max(1);
        self.best_limit = self.best_limit.min(ceiling).max(1);
        self.testing_from = None;
        self.last_lower = None;
        self.settled = self.effective_limit == ceiling;
        self.best_score = None;
        self.last_probe = now;
        self.reset_window(now, 0, completed_total);
        self.reason = "pool ceiling changed; restarting calibration".into();
    }

    fn observe(
        &mut self,
        now: Instant,
        active: u32,
        completed_total: u64,
        gpu: Option<GpuSample>,
        host_cpu_percent: Option<f64>,
    ) {
        let telemetry_recovered = self.telemetry_failed_open && gpu.is_some();
        self.latest_gpu = gpu;
        self.latest_host_cpu_percent = host_cpu_percent;

        let Some(gpu) = gpu else {
            // Fail open: telemetry loss must never strand work behind a stale
            // learned cap. A fixed maxActive still applies as the ceiling.
            self.effective_limit = self.ceiling;
            self.best_limit = self.ceiling;
            self.best_score = None;
            self.testing_from = None;
            self.last_lower = None;
            self.settled = true;
            self.telemetry_failed_open = true;
            self.reason = "GPU telemetry unavailable; using full residency".into();
            self.reset_window(now, active, completed_total);
            return;
        };

        if telemetry_recovered {
            self.effective_limit = Self::initial_limit(self.ceiling);
            self.best_limit = self.effective_limit;
            self.best_score = None;
            self.testing_from = None;
            self.last_lower = None;
            self.settled = self.effective_limit == self.ceiling;
            self.last_probe = now;
            self.telemetry_failed_open = false;
            self.reason = "GPU telemetry recovered; restarting calibration".into();
            self.reset_window(now, active, completed_total);
            return;
        }

        if active != self.observed_active {
            self.reset_window(now, active, completed_total);
        }
        self.window_samples = self.window_samples.saturating_add(1);
        self.gpu_utilization_sum += gpu.utilization_percent;
        self.host_cpu_sum += host_cpu_percent.unwrap_or(0.0);

        let elapsed = now.saturating_duration_since(self.window_started);
        if active > self.effective_limit {
            self.reason = format!(
                "waiting for active leases to drain from {active} to {}",
                self.effective_limit
            );
            return;
        }
        let at_requested_residency = active == self.effective_limit;
        if !at_requested_residency
            || self.window_samples < MIN_STABLE_SAMPLES
            || elapsed < OBSERVATION_WINDOW
        {
            self.reason = format!(
                "observing stable residency at {active}/{}",
                self.effective_limit
            );
            return;
        }

        let mean_gpu = self.gpu_utilization_sum / f64::from(self.window_samples);
        let mean_cpu = self.host_cpu_sum / f64::from(self.window_samples);
        let completed = completed_total.saturating_sub(self.completed_at_start);
        let completion_rate = (completed > 0 && elapsed.as_secs_f64() > 0.0)
            .then(|| completed as f64 / elapsed.as_secs_f64());
        let score = Score {
            gpu_utilization_percent: mean_gpu,
            completion_rate,
        };
        let memory_safe = gpu.memory_safe();
        let cpu_safe = mean_cpu < CPU_SATURATION_PERCENT;

        if let Some((prior_limit, prior_score)) = self.testing_from.take() {
            let improvement = score.improvement_over(prior_score);
            if improvement < MARGINAL_GAIN_PERCENT || !memory_safe || !cpu_safe {
                self.effective_limit = prior_limit;
                self.best_limit = prior_limit;
                self.best_score = Some(prior_score);
                self.settled = true;
                self.last_probe = now;
                self.reason = if !memory_safe {
                    format!(
                        "returned to {prior_limit}: limiting GPU has {} MiB free; preserving {} MiB reserve",
                        gpu.limiting_device_free_memory_mib,
                        gpu.limiting_device_reserve_mib
                    )
                } else if !cpu_safe {
                    format!("returned to {prior_limit}: host CPU saturated at {mean_cpu:.1}%")
                } else {
                    format!(
                        "returned to {prior_limit}: marginal gain {improvement:.1}% is below {MARGINAL_GAIN_PERCENT:.1}%"
                    )
                };
                self.reset_window(now, active, completed_total);
                return;
            }
            self.last_lower = Some((prior_limit, prior_score));
        }

        self.best_limit = self.effective_limit;
        self.best_score = Some(score);

        if self.settled
            && now.saturating_duration_since(self.last_probe) >= REPROBE_INTERVAL
            && self.has_pressure(now)
            && self.effective_limit < self.ceiling
        {
            self.settled = false;
        }

        if self.settled && (!memory_safe || !cpu_safe) {
            if let Some((lower_limit, lower_score)) = self.last_lower.take() {
                self.effective_limit = lower_limit;
                self.best_limit = lower_limit;
                self.best_score = Some(lower_score);
                self.last_probe = now;
                self.reason = if !memory_safe {
                    format!(
                        "returned to {lower_limit}: limiting GPU has {} MiB free; preserving {} MiB reserve",
                        gpu.limiting_device_free_memory_mib,
                        gpu.limiting_device_reserve_mib
                    )
                } else {
                    format!("returned to {lower_limit}: host CPU saturated at {mean_cpu:.1}%")
                };
                self.reset_window(now, active, completed_total);
                return;
            }
        }

        if !self.settled && self.has_pressure(now) && memory_safe && cpu_safe {
            let next = self.effective_limit.saturating_mul(2).min(self.ceiling);
            if next > self.effective_limit {
                self.testing_from = Some((self.effective_limit, score));
                self.effective_limit = next;
                self.reason = format!("probing resident limit {next}");
                self.reset_window(now, active, completed_total);
                return;
            }
        }

        self.settled = true;
        self.last_probe = now;
        self.reason = if !memory_safe {
            format!(
                "holding at {}: limiting GPU has {} MiB free; preserving {} MiB reserve",
                self.effective_limit,
                gpu.limiting_device_free_memory_mib,
                gpu.limiting_device_reserve_mib
            )
        } else if !cpu_safe {
            format!(
                "holding at {} because host CPU is {:.1}% busy",
                self.effective_limit, mean_cpu
            )
        } else {
            format!("settled at resident limit {}", self.effective_limit)
        };
        self.reset_window(now, active, completed_total);
    }

    fn snapshot(&self) -> AdmissionSnapshot {
        AdmissionSnapshot {
            effective_limit: self.effective_limit,
            reason: self.reason.clone(),
            calibrating: !self.settled,
            gpu_utilization_percent: self.latest_gpu.map(|sample| sample.utilization_percent),
            gpu_memory_used_mib: self.latest_gpu.map(|sample| sample.used_memory_mib),
            gpu_memory_total_mib: self.latest_gpu.map(|sample| sample.total_memory_mib),
            host_cpu_percent: self.latest_host_cpu_percent,
        }
    }
}

/// Thread-safe bridge between the HTTP admission path and the background
/// telemetry loop.
#[derive(Default)]
pub struct AdmissionRegistry {
    pools: Mutex<HashMap<String, PoolAdmissionState>>,
}

impl AdmissionRegistry {
    fn ceiling(pool: &ForkPoolRecord) -> u32 {
        pool.max_active
            .unwrap_or(pool.desired_ready)
            .min(pool.desired_ready)
            .max(1)
    }

    /// Record that a caller had work but could not claim another resident slot.
    pub fn note_blocked(&self, pool_name: &str) {
        if let Some(state) = self.pools.lock().get_mut(pool_name) {
            state.last_blocked = Some(Instant::now());
        }
    }

    /// Current dynamic claim limit. `None` means the pool uses its existing
    /// static `maxActive`/ready-slot behavior.
    pub fn limit(&self, pool: &ForkPoolRecord) -> Option<u32> {
        if !pool.auto_admission {
            return None;
        }
        self.pools
            .lock()
            .get(&pool.name)
            .map(|state| state.effective_limit)
    }

    /// Return the latest published telemetry and decision for an automatic pool.
    pub fn snapshot(&self, pool: &ForkPoolRecord) -> Option<AdmissionSnapshot> {
        if !pool.auto_admission {
            return None;
        }
        self.pools
            .lock()
            .get(&pool.name)
            .map(PoolAdmissionState::snapshot)
    }

    /// Feed one controller-tick observation into a pool's calibration state.
    pub fn observe(
        &self,
        pool: &ForkPoolRecord,
        active: u32,
        completed_total: u64,
        gpu: Option<GpuSample>,
        host_cpu_percent: Option<f64>,
    ) {
        if !pool.auto_admission {
            self.pools.lock().remove(&pool.name);
            return;
        }
        self.observe_at(
            pool,
            active,
            completed_total,
            gpu,
            host_cpu_percent,
            Instant::now(),
        );
    }

    fn observe_at(
        &self,
        pool: &ForkPoolRecord,
        active: u32,
        completed_total: u64,
        gpu: Option<GpuSample>,
        host_cpu_percent: Option<f64>,
        now: Instant,
    ) {
        let ceiling = Self::ceiling(pool);
        let mut pools = self.pools.lock();
        let state = pools
            .entry(pool.name.clone())
            .or_insert_with(|| PoolAdmissionState::new(ceiling, now, completed_total));
        state.update_ceiling(ceiling, now, completed_total);
        state.observe(now, active, completed_total, gpu, host_cpu_percent);
    }

    /// Forget runtime state for pools that no longer exist.
    pub fn retain_pools(&self, names: &HashSet<String>) {
        self.pools.lock().retain(|name, _| names.contains(name));
    }
}

/// Samples aggregate host CPU busy time from `/proc/stat`.
#[derive(Default)]
pub struct HostCpuSampler {
    #[cfg(target_os = "linux")]
    previous: Option<(u64, u64)>,
}

impl HostCpuSampler {
    /// Sample host CPU busy percentage since the previous call.
    pub fn sample(&mut self) -> Option<f64> {
        #[cfg(target_os = "linux")]
        {
            let data = std::fs::read_to_string("/proc/stat").ok()?;
            let line = data.lines().next()?;
            let mut fields = line.split_whitespace();
            if fields.next()? != "cpu" {
                return None;
            }
            let values: Vec<u64> = fields.filter_map(|field| field.parse().ok()).collect();
            if values.len() < 4 {
                return None;
            }
            let total = values.iter().copied().sum::<u64>();
            let idle = values[3].saturating_add(values.get(4).copied().unwrap_or(0));
            let result = self.previous.and_then(|(old_total, old_idle)| {
                let delta_total = total.saturating_sub(old_total);
                let delta_idle = idle.saturating_sub(old_idle);
                (delta_total > 0).then(|| {
                    100.0 * (delta_total.saturating_sub(delta_idle)) as f64 / delta_total as f64
                })
            });
            self.previous = Some((total, idle));
            result
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }
}

/// Small dynamically-loaded NVML surface. The host NVIDIA driver already ships
/// this library; keeping it optional preserves CPU-only and non-Linux builds.
#[cfg(target_os = "linux")]
pub struct NvmlSampler {
    _library: libloading::Library,
    shutdown: unsafe extern "C" fn() -> i32,
    device_count: unsafe extern "C" fn(*mut u32) -> i32,
    device_handle: unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32,
    memory_info: unsafe extern "C" fn(*mut std::ffi::c_void, *mut NvmlMemory) -> i32,
    utilization: unsafe extern "C" fn(*mut std::ffi::c_void, *mut NvmlUtilization) -> i32,
}

#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Default)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[cfg(target_os = "linux")]
impl NvmlSampler {
    /// Load and initialize the host driver's NVML library.
    pub fn new() -> Result<Self, String> {
        type Init = unsafe extern "C" fn() -> i32;
        type Shutdown = unsafe extern "C" fn() -> i32;
        type DeviceCount = unsafe extern "C" fn(*mut u32) -> i32;
        type DeviceHandle = unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32;
        type MemoryInfo = unsafe extern "C" fn(*mut std::ffi::c_void, *mut NvmlMemory) -> i32;
        type Utilization = unsafe extern "C" fn(*mut std::ffi::c_void, *mut NvmlUtilization) -> i32;

        // SAFETY: the copied function pointers remain valid because the Library
        // is retained in the returned sampler until after `shutdown` runs.
        unsafe {
            let library = libloading::Library::new("libnvidia-ml.so.1")
                .map_err(|error| format!("load libnvidia-ml.so.1: {error}"))?;
            let init = *library
                .get::<Init>(b"nvmlInit_v2\0")
                .map_err(|error| format!("resolve nvmlInit_v2: {error}"))?;
            let shutdown = *library
                .get::<Shutdown>(b"nvmlShutdown\0")
                .map_err(|error| format!("resolve nvmlShutdown: {error}"))?;
            let device_count = *library
                .get::<DeviceCount>(b"nvmlDeviceGetCount_v2\0")
                .map_err(|error| format!("resolve nvmlDeviceGetCount_v2: {error}"))?;
            let device_handle = *library
                .get::<DeviceHandle>(b"nvmlDeviceGetHandleByIndex_v2\0")
                .map_err(|error| format!("resolve nvmlDeviceGetHandleByIndex_v2: {error}"))?;
            let memory_info = *library
                .get::<MemoryInfo>(b"nvmlDeviceGetMemoryInfo\0")
                .map_err(|error| format!("resolve nvmlDeviceGetMemoryInfo: {error}"))?;
            let utilization = *library
                .get::<Utilization>(b"nvmlDeviceGetUtilizationRates\0")
                .map_err(|error| format!("resolve nvmlDeviceGetUtilizationRates: {error}"))?;
            let status = init();
            if status != 0 {
                return Err(format!("nvmlInit_v2 returned {status}"));
            }
            Ok(Self {
                _library: library,
                shutdown,
                device_count,
                device_handle,
                memory_info,
                utilization,
            })
        }
    }

    /// Sample aggregate utilization and per-device memory safety across NVIDIA devices.
    pub fn sample(&mut self) -> Option<GpuSample> {
        // SAFETY: NVML owns device handles; all output pointers target valid,
        // initialized storage with the ABI layouts documented by NVML.
        unsafe {
            let mut count = 0;
            if (self.device_count)(&mut count) != 0 || count == 0 {
                return None;
            }
            let mut devices = Vec::with_capacity(count as usize);
            for index in 0..count {
                let mut device = std::ptr::null_mut();
                if (self.device_handle)(index, &mut device) != 0 || device.is_null() {
                    return None;
                }
                let mut memory = NvmlMemory::default();
                let mut utilization = NvmlUtilization::default();
                if (self.memory_info)(device, &mut memory) != 0
                    || (self.utilization)(device, &mut utilization) != 0
                {
                    return None;
                }
                devices.push(GpuDeviceSample {
                    utilization_percent: f64::from(utilization.gpu),
                    used_memory_mib: memory.used / (1024 * 1024),
                    free_memory_mib: memory.free / (1024 * 1024),
                    total_memory_mib: memory.total / (1024 * 1024),
                });
            }
            GpuSample::from_devices(devices)
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for NvmlSampler {
    fn drop(&mut self) {
        // SAFETY: paired with the successful nvmlInit_v2 call in `new`.
        unsafe {
            (self.shutdown)();
        }
    }
}

#[cfg(not(target_os = "linux"))]
/// Reports unavailable NVML telemetry on unsupported hosts.
pub struct NvmlSampler;

#[cfg(not(target_os = "linux"))]
impl NvmlSampler {
    /// Report that NVML telemetry is unsupported on this host.
    pub fn new() -> Result<Self, String> {
        Err("NVML admission telemetry is only available on Linux".into())
    }

    /// Return no GPU sample on unsupported hosts.
    pub fn sample(&mut self) -> Option<GpuSample> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(size: u32) -> ForkPoolRecord {
        ForkPoolRecord {
            name: "rollouts".into(),
            golden: "golden".into(),
            desired_ready: size,
            max_active: None,
            auto_admission: true,
            share_weights: true,
            ready_timeout_secs: 240,
            lease_ttl_secs: 300,
            created_at: 1,
            deleting: false,
        }
    }

    fn gpu(utilization: f64, used: u64) -> Option<GpuSample> {
        GpuSample::from_devices([GpuDeviceSample {
            utilization_percent: utilization,
            used_memory_mib: used,
            free_memory_mib: 80 * 1024 - used,
            total_memory_mib: 80 * 1024,
        }])
    }

    fn stable_window(
        registry: &AdmissionRegistry,
        pool: &ForkPoolRecord,
        start: Instant,
        observation: (u32, u64, f64, u64, f64),
    ) {
        let (active, completed, utilization, used, cpu) = observation;
        for second in 0..=8 {
            registry.observe_at(
                pool,
                active,
                completed,
                gpu(utilization, used),
                Some(cpu),
                start + Duration::from_secs(second),
            );
        }
    }

    fn multi_gpu_sample(devices: &[(f64, u64, u64, u64)]) -> GpuSample {
        GpuSample::from_devices(devices.iter().map(
            |&(utilization_percent, used_memory_mib, free_memory_mib, total_memory_mib)| {
                GpuDeviceSample {
                    utilization_percent,
                    used_memory_mib,
                    free_memory_mib,
                    total_memory_mib,
                }
            },
        ))
        .unwrap()
    }

    #[test]
    fn imbalanced_multi_gpu_memory_uses_the_limiting_device() {
        let sample = multi_gpu_sample(&[
            (95.0, 75 * 1024, 5 * 1024, 80 * 1024),
            (5.0, 0, 80 * 1024, 80 * 1024),
        ]);

        assert_eq!(sample.used_memory_mib, 75 * 1024);
        assert_eq!(sample.total_memory_mib, 160 * 1024);
        assert!(sample.total_memory_mib - sample.used_memory_mib > 16 * 1024);
        assert_eq!(sample.limiting_device_free_memory_mib, 5 * 1024);
        assert_eq!(sample.limiting_device_reserve_mib, 8 * 1024);
        assert!(!sample.memory_safe());
    }

    #[test]
    fn heterogeneous_devices_each_use_their_own_reserve() {
        let sample = multi_gpu_sample(&[
            (70.0, 15 * 1024, 9 * 1024, 24 * 1024),
            (80.0, 135 * 1024, 25 * 1024, 160 * 1024),
        ]);

        assert_eq!(sample.limiting_device_free_memory_mib, 9 * 1024);
        assert_eq!(sample.limiting_device_reserve_mib, 8 * 1024);
        assert!(sample.memory_safe());
    }

    #[test]
    fn per_device_pressure_rejects_a_larger_candidate() {
        let registry = AdmissionRegistry::default();
        let pool = pool(12);
        let start = Instant::now();
        registry.observe_at(&pool, 0, 0, gpu(0.0, 10_000), Some(10.0), start);
        registry.note_blocked("rollouts");
        stable_window(&registry, &pool, start, (4, 0, 50.0, 30_000, 50.0));
        assert_eq!(registry.limit(&pool), Some(8));

        let imbalanced = multi_gpu_sample(&[
            (95.0, 75 * 1024, 5 * 1024, 80 * 1024),
            (5.0, 0, 80 * 1024, 80 * 1024),
        ]);
        for second in 0..=8 {
            registry.observe_at(
                &pool,
                8,
                0,
                Some(imbalanced),
                Some(50.0),
                start + Duration::from_secs(9 + second),
            );
        }

        assert_eq!(registry.limit(&pool), Some(4));
        let reason = registry.snapshot(&pool).unwrap().reason;
        assert!(
            reason.contains("limiting GPU has 5120 MiB free"),
            "{reason}"
        );
    }

    #[test]
    fn calibrates_to_the_measured_pareto_point_without_card_specific_limit() {
        let registry = AdmissionRegistry::default();
        let pool = pool(12);
        let start = Instant::now();
        registry.observe_at(&pool, 0, 0, gpu(0.0, 10_000), Some(10.0), start);
        registry.note_blocked("rollouts");

        stable_window(&registry, &pool, start, (4, 0, 56.6, 30_828, 62.0));
        assert_eq!(registry.limit(&pool), Some(8));

        stable_window(
            &registry,
            &pool,
            start + Duration::from_secs(9),
            (8, 0, 90.2, 51_138, 62.0),
        );
        assert_eq!(registry.limit(&pool), Some(12));

        stable_window(
            &registry,
            &pool,
            start + Duration::from_secs(18),
            (12, 0, 88.0, 71_445, 62.0),
        );
        assert_eq!(registry.limit(&pool), Some(8));
        assert!(!registry.snapshot(&pool).unwrap().calibrating);
    }

    #[test]
    fn telemetry_failure_falls_back_to_full_residency() {
        let registry = AdmissionRegistry::default();
        let pool = pool(12);
        registry.observe_at(&pool, 0, 0, None, None, Instant::now());
        let snapshot = registry.snapshot(&pool).unwrap();
        assert_eq!(snapshot.effective_limit, 12);
        assert!(snapshot.reason.contains("telemetry unavailable"));
    }

    #[test]
    fn telemetry_recovery_restarts_calibration_after_existing_leases_drain() {
        let registry = AdmissionRegistry::default();
        let pool = pool(12);
        let start = Instant::now();
        registry.observe_at(&pool, 12, 0, None, None, start);
        assert_eq!(registry.limit(&pool), Some(12));

        registry.observe_at(
            &pool,
            12,
            0,
            gpu(90.0, 70_000),
            Some(50.0),
            start + Duration::from_secs(1),
        );
        assert_eq!(registry.limit(&pool), Some(4));
        assert!(registry
            .snapshot(&pool)
            .unwrap()
            .reason
            .contains("restarting calibration"));

        registry.observe_at(
            &pool,
            12,
            0,
            gpu(90.0, 70_000),
            Some(50.0),
            start + Duration::from_secs(2),
        );
        assert!(registry.snapshot(&pool).unwrap().reason.contains("drain"));
        assert_eq!(registry.limit(&pool), Some(4));
    }

    #[test]
    fn cpu_saturation_rejects_a_larger_candidate() {
        let registry = AdmissionRegistry::default();
        let pool = pool(12);
        let start = Instant::now();
        registry.observe_at(&pool, 0, 0, gpu(0.0, 10_000), Some(10.0), start);
        registry.note_blocked("rollouts");
        stable_window(&registry, &pool, start, (4, 0, 50.0, 30_000, 50.0));
        assert_eq!(registry.limit(&pool), Some(8));
        stable_window(
            &registry,
            &pool,
            start + Duration::from_secs(9),
            (8, 0, 90.0, 50_000, 96.0),
        );
        assert_eq!(registry.limit(&pool), Some(4));
    }

    #[test]
    fn late_cpu_saturation_rolls_back_the_last_accepted_candidate() {
        let registry = AdmissionRegistry::default();
        let pool = pool(12);
        let start = Instant::now();
        registry.observe_at(&pool, 0, 0, gpu(0.0, 10_000), Some(10.0), start);
        registry.note_blocked("rollouts");
        stable_window(&registry, &pool, start, (4, 0, 50.0, 30_000, 50.0));
        stable_window(
            &registry,
            &pool,
            start + Duration::from_secs(9),
            (8, 0, 75.0, 50_000, 70.0),
        );
        assert_eq!(registry.limit(&pool), Some(12));
        stable_window(
            &registry,
            &pool,
            start + Duration::from_secs(18),
            (12, 0, 95.0, 70_000, 80.0),
        );
        assert_eq!(registry.limit(&pool), Some(12));
        stable_window(
            &registry,
            &pool,
            start + Duration::from_secs(27),
            (12, 0, 95.0, 71_000, 95.0),
        );
        assert_eq!(registry.limit(&pool), Some(8));
        let snapshot = registry.snapshot(&pool).unwrap();
        assert!(snapshot.reason.contains("drain"), "{}", snapshot.reason);
    }

    #[test]
    fn a_static_pool_never_receives_a_dynamic_limit() {
        let registry = AdmissionRegistry::default();
        let mut pool = pool(12);
        pool.auto_admission = false;
        registry.observe(&pool, 0, 0, gpu(0.0, 0), Some(0.0));
        assert_eq!(registry.limit(&pool), None);
        assert_eq!(registry.snapshot(&pool), None);
    }
}
