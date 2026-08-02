//! Content-addressed OCI image store shared by the local CLI and the cloud
//! worker path — one implementation so both behave identically (issue #756).
//!
//! Layers are deduplicated by digest across every caller, but ACCESS is
//! authorized per caller on every call, including cache hits: [`ensure_image`]
//! resolves the manifest with the caller's credentials first, so the registry's
//! own `repository:<repo>:pull` authorization decides whether the caller may see
//! the (possibly already-cached) bytes. A cached digest is never served to a
//! caller who cannot pull it.
//!
//! [`ensure_image`]: ImageStore::ensure_image

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::registry::{registry_client, PullAuth, Reference};
use crate::{Error, Result};

/// The bits of an OCI image config the launcher needs to boot a cached image:
/// the process to run, its environment, and working directory. Cached beside the
/// layers (as `config.json`) so a warm run needs no network at all.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ImageConfig {
    /// The image's `Entrypoint` (prepended to `cmd` / the run command).
    pub entrypoint: Vec<String>,
    /// The image's default `Cmd` (used when the run supplies no command).
    pub cmd: Vec<String>,
    /// The image's `Env` (`KEY=VALUE` strings).
    pub env: Vec<String>,
    /// The image's `WorkingDir` (empty string if unset).
    pub workdir: String,
}

/// A ready-to-boot cache entry: the extracted overlay lowerdirs plus the image's
/// config (entrypoint/cmd/env/workdir). Returned by [`ImageStore::ensure_image`].
pub struct CachedImage {
    /// The `layers/` dir to mount read-only as the overlay lowerdirs.
    pub layers: PathBuf,
    /// The image process configuration to boot with.
    pub config: ImageConfig,
}

/// The `config` object inside an OCI image config blob. Fields are capitalized in
/// the OCI spec and each may be absent/null, hence the `Option`s.
#[derive(serde::Deserialize, Default)]
struct OciConfigInner {
    #[serde(rename = "Entrypoint")]
    entrypoint: Option<Vec<String>>,
    #[serde(rename = "Cmd")]
    cmd: Option<Vec<String>>,
    #[serde(rename = "Env")]
    env: Option<Vec<String>>,
    #[serde(rename = "WorkingDir")]
    working_dir: Option<String>,
}

/// The top level of an OCI image config blob (only its `config` object matters).
#[derive(serde::Deserialize, Default)]
struct OciConfigBlob {
    #[serde(default)]
    config: OciConfigInner,
}

impl From<OciConfigBlob> for ImageConfig {
    fn from(b: OciConfigBlob) -> Self {
        ImageConfig {
            entrypoint: b.config.entrypoint.unwrap_or_default(),
            cmd: b.config.cmd.unwrap_or_default(),
            env: b.config.env.unwrap_or_default(),
            workdir: b.config.working_dir.unwrap_or_default(),
        }
    }
}

/// A content-addressed OCI image store rooted at a shared directory. On a node
/// this is the same `_shared` tree used by packs, so images and packs dedup in
/// one place and mount through the identical idmapped-bind path.
pub struct ImageStore {
    root: PathBuf,
}

impl ImageStore {
    /// Store rooted at `root`; each image lives at `root/<manifest-digest>/`.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// The node/host content-addressed root (`<vm_cache_root>/_shared`), shared
    /// with the pack store so identical layers are stored once for everyone.
    pub fn shared() -> Self {
        Self::new(crate::agent::shared_pack_cache_root())
    }

    /// Resolve, authorize, and ensure `reference`'s layers are in the store,
    /// returning the `layers/` dir to idmap-mount into the VM.
    ///
    /// The auth gate (manifest resolution with `auth`) runs on every call, before
    /// the cache is consulted, so a cache hit cannot leak an image the caller is
    /// not authorized to pull.
    pub async fn ensure_image(&self, reference: &str, auth: &PullAuth) -> Result<CachedImage> {
        // ── AUTH GATE + content address ─────────────────────────────────────
        // Resolve + authorize the manifest with the caller's credentials, across
        // the candidate registries the guest would try. Runs before the cache is
        // consulted, so a hit cannot leak an image the caller can't pull.
        let r = resolve_authorized(reference, auth).await?;
        // The key binds the manifest digest to the REGISTRY + REPO it was
        // authorized against (see `entry_key`), so passing the gate at one
        // registry never unlocks an entry filled from another.
        let key = entry_key(&r.registry, &r.repo, &r.digest);
        let entry = self.root.join(&key);
        let layers = entry.join("layers");

        // ── CACHE HIT (already authorized above) ───────────────────────────
        if is_intact(&entry) {
            touch(&entry);
            return Ok(CachedImage {
                config: read_config(&entry)?,
                layers,
            });
        }

        // ── MISS ────────────────────────────────────────────────────────────
        // Serialize fills of the SAME key across processes AND tasks with an
        // exclusive lock, mirroring the pack store. Without it two fillers share
        // a staging dir: one wipes the other's extracted layers and then
        // publishes an entry whose `layer-order` lists dirs that do not exist.
        std::fs::create_dir_all(&self.root)
            .map_err(|e| Error::config("image-store", e.to_string()))?;
        let _guard = FillLock::acquire(&self.root.join(format!(".lock-{key}")))?;

        // Re-check under the lock: whoever held it before us may have filled it.
        if is_intact(&entry) {
            touch(&entry);
            return Ok(CachedImage {
                config: read_config(&entry)?,
                layers,
            });
        }

        let manifest: smolvm_registry::OciManifest = serde_json::from_slice(&r.manifest_bytes)
            .map_err(|e| Error::agent("image-store: parse manifest", e.to_string()))?;
        // Unique per fill (pid + nanos): the lock already excludes concurrent
        // fillers, and uniqueness keeps a crashed fill's debris from being
        // mistaken for ours. `sweep_staging` reaps whatever a crash leaves.
        let staging = self.root.join(format!(".staging-{key}-{}", fill_nonce()));
        let _ = std::fs::remove_dir_all(&staging);
        let config = materialize_entry(&r.client, &r.repo, &manifest, &staging)
            .await
            .inspect_err(|_| {
                let _ = std::fs::remove_dir_all(&staging);
            })?;

        // Atomic publish. A pre-existing directory here is debris (a crash, or a
        // partially removed entry): rename would fail with ENOTEMPTY forever, so
        // clear it and retry once rather than poisoning the key permanently.
        match std::fs::rename(&staging, &entry) {
            Ok(()) => {}
            Err(_) if is_intact(&entry) => {
                // A concurrent caller published the same key first — fine.
                let _ = std::fs::remove_dir_all(&staging);
            }
            Err(_) if entry.exists() => {
                let _ = std::fs::remove_dir_all(&entry);
                std::fs::rename(&staging, &entry).map_err(|e| {
                    let _ = std::fs::remove_dir_all(&staging);
                    Error::config("image-store: publish", e.to_string())
                })?;
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(Error::config("image-store: publish", e.to_string()));
            }
        }

        // Bound the store AFTER publishing, never evicting what we just made.
        sweep_staging(&self.root);
        prune_store(&self.root, image_cache_max_bytes(), &entry);
        Ok(CachedImage { layers, config })
    }
}

/// Environment switch that routes a registry-image machine through the host-side
/// store: the host pulls and extracts the image once into content-addressed
/// overlay lowerdirs, and every later boot mounts them read-only instead of
/// pulling and flattening the image inside each VM.
///
/// Opt-in while the path is being proven on Linux; the default is unchanged.
const IMAGE_STORE_ENV: &str = "SMOLVM_IMAGE_STORE";

/// Whether the host-side image store is enabled for this process.
pub fn image_store_enabled() -> bool {
    std::env::var(IMAGE_STORE_ENV).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Fill (or reuse) the store for `reference` and return the ready-to-boot entry,
/// from a synchronous caller.
///
/// The VM start paths are synchronous, so this drives the async fill on a
/// short-lived runtime. Extraction is Linux+root (whiteout translation needs
/// `mknod`/`setxattr`), which is where the node and the `--oci-cache` Linux path
/// run; on other hosts the caller keeps its existing behavior.
pub fn ensure_image_blocking(reference: &str, auth: &PullAuth) -> Result<CachedImage> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| Error::config("image-store: runtime", e.to_string()))?;
    rt.block_on(ImageStore::shared().ensure_image(reference, auth))
}

/// The cached image config filename inside an entry (sibling of `layers/`).
const CONFIG_FILE: &str = "config.json";

/// Read the cached [`ImageConfig`] from an intact entry.
fn read_config(entry: &Path) -> Result<ImageConfig> {
    let bytes = std::fs::read(entry.join(CONFIG_FILE))
        .map_err(|e| Error::config("image-store: read config", e.to_string()))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::agent("image-store: parse cached config", e.to_string()))
}

/// The outcome of the auth gate: the authorized client + repo it succeeded
/// against, the manifest bytes, and their digest (the content-address key).
struct Resolved {
    client: smolvm_registry::RegistryClient,
    /// The registry host the manifest was actually authorized against. Part of
    /// the cache key so an entry is only ever served to a caller who passed the
    /// gate at the SAME registry.
    registry: String,
    repo: String,
    manifest_bytes: Vec<u8>,
    digest: String,
}

/// Resolve + authorize `reference` across the candidate registries the guest
/// would try, returning the winning client, repo, manifest, and content digest.
///
/// The registry authorizes `repository:<repo>:pull` for the caller's credentials
/// here; an unauthorized caller is rejected (401/403). This is the single gate
/// behind both [`authorized_digest`] and [`ImageStore::ensure_image`].
async fn resolve_authorized(reference: &str, auth: &PullAuth) -> Result<Resolved> {
    let parsed = Reference::parse(reference)
        .map_err(|e| Error::config("image-auth", format!("bad reference: {}", e.reason)))?;
    let want = parsed
        .digest
        .clone()
        .or_else(|| parsed.tag.clone())
        .unwrap_or_else(|| "latest".to_string());
    // OCI-image credentials live under `images` (docker.io/ghcr/...), the same
    // config the guest pull consults.
    let config = crate::SmolSettings::load()?.images;

    // Resolve the registry the way the GUEST does — via `registry_pull_hosts`, not
    // the configured default — so the host-side gate targets the same registry the
    // in-guest pull would (a bare `alpine` is Docker Hub, not the smol registry).
    let mut last_err: Option<String> = None;
    for host in &crate::registry::registry_pull_hosts(reference) {
        let client = registry_client(host, &config, auth);
        let repo = repo_for(host, &parsed);
        match client.get_manifest_resolved(&repo, &want).await {
            Ok(manifest_bytes) => {
                let digest = format!("sha256:{}", hex::encode(Sha256::digest(&manifest_bytes)));
                return Ok(Resolved {
                    client,
                    registry: host.clone(),
                    repo,
                    manifest_bytes,
                    digest,
                });
            }
            // Keep the FIRST failure. `registry_pull_hosts` is a DNS allow-list,
            // not a list of real endpoints — Docker Hub yields
            // ["docker.io", "docker.com"], and letting the later marketing-host
            // failure overwrite the real 401/429 from docker.io would surface a
            // nonsense error to the user.
            Err(e) => {
                let _ = last_err.get_or_insert_with(|| e.to_string());
            }
        }
    }
    Err(Error::agent(
        "image-store: authorize",
        last_err.unwrap_or_else(|| "no candidate registry resolved the image".to_string()),
    ))
}

/// The auth gate on its own: resolve `reference` with the caller's credentials and
/// return the content digest. A caller who cannot pull the repo is rejected here.
/// Used by paths that materialize the image through the proven pack pipeline but
/// still need the per-caller authorization and the content address as their key.
pub async fn authorized_digest(reference: &str, auth: &PullAuth) -> Result<String> {
    Ok(resolve_authorized(reference, auth).await?.digest)
}

/// The repository path for a reference against a candidate `host`. Docker Hub
/// official images (no namespace) live under the implicit `library/` namespace,
/// so a bare `alpine` becomes `library/alpine` — without which the pull-scope is
/// wrong and the registry answers 401.
fn repo_for(host: &str, r: &Reference) -> String {
    let docker_hub = matches!(
        host,
        "docker.io" | "docker.com" | "index.docker.io" | "registry-1.docker.io"
    );
    match &r.namespace {
        Some(ns) => format!("{}/{}", ns, r.name),
        None if docker_hub => format!("library/{}", r.name),
        None => r.name.clone(),
    }
}

/// A filesystem-safe directory name for a digest (`sha256:abc` → `sha256-abc`).
fn digest_dir(digest: &str) -> String {
    digest.replace(':', "-")
}

/// Stream a blob to `dest`, hashing as it goes, and reject it unless the content
/// hashes to `digest`. Streaming (rather than `pull_blob`) removes both the
/// 64 MiB body cap and the full-blob memory buffer; verifying from the streamed
/// bytes keeps the content-addressing guarantee — the file is only accepted if
/// it hashes to the digest the authorized manifest named.
async fn stream_blob_verified(
    client: &smolvm_registry::RegistryClient,
    repo: &str,
    digest: &str,
    dest: &Path,
) -> Result<()> {
    use futures_util::StreamExt;
    use std::io::Write;

    let mut stream = client
        .pull_blob_stream(repo, digest)
        .await
        .map_err(|e| Error::agent("image-store: pull layer", e.to_string()))?;
    let file =
        std::fs::File::create(dest).map_err(|e| Error::config("image-store", e.to_string()))?;
    let mut writer = std::io::BufWriter::new(file);
    let mut hasher = Sha256::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Error::agent("image-store: pull layer", e.to_string()))?;
        hasher.update(&chunk);
        writer
            .write_all(&chunk)
            .map_err(|e| Error::config("image-store: write layer", e.to_string()))?;
    }
    writer
        .flush()
        .map_err(|e| Error::config("image-store: write layer", e.to_string()))?;

    let got = format!("sha256:{}", hex::encode(hasher.finalize()));
    if got != digest {
        let _ = std::fs::remove_file(dest);
        return Err(Error::agent(
            "image-store: layer digest",
            format!("blob content mismatch: expected {digest}, got {got}"),
        ));
    }
    Ok(())
}

/// The cache key for an image: the manifest digest bound to the registry and
/// repository it was authorized against.
///
/// Keying on the manifest digest ALONE would make the auth gate bypassable.
/// `resolve_authorized` accepts the first candidate registry that answers 200 —
/// including one the caller controls. Anyone holding a private image's manifest
/// bytes (manifests leak far more easily than blobs: an old public tag, a CI
/// log, `docker manifest inspect`) could serve those exact bytes from their own
/// registry, pass the gate there, compute the identical digest, and be handed
/// the private image's extracted layers without ever authenticating to the real
/// registry. Binding the key to (registry, repo) means passing the gate at one
/// registry can only ever unlock content filled from that same registry.
///
/// Dedup is preserved where it matters: every tenant pulling
/// `docker.io/library/alpine` at the same digest still shares one entry.
fn entry_key(registry: &str, repo: &str, manifest_digest: &str) -> String {
    let scoped = format!("{registry}/{repo}@{manifest_digest}");
    format!("sha256-{}", hex::encode(Sha256::digest(scoped.as_bytes())))
}

/// Environment override (bytes) for the maximum on-disk size of the image store.
const IMAGE_CACHE_MAX_BYTES_ENV: &str = "SMOLVM_IMAGE_CACHE_MAX_BYTES";

/// Size ceiling for the extracted-image store, default 20 GiB.
///
/// EXTRACTED layers are substantially larger than the compressed blobs they came
/// from (measured: `node:20` is 379 MiB compressed, 636 MiB extracted), so an
/// unbounded store fills a node's disk and every tenant on it then fails to
/// start. The default is deliberately larger than the init-bake cache's 10 GiB
/// for that reason.
fn image_cache_max_bytes() -> u64 {
    std::env::var(IMAGE_CACHE_MAX_BYTES_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20 * 1024 * 1024 * 1024)
}

/// Mark an entry as recently used, so LRU eviction keeps hot images.
fn touch(entry: &Path) {
    if let Ok(f) = std::fs::File::open(entry) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

/// A unique-per-fill suffix so a crashed fill's debris is never confused with an
/// in-progress one. (Uniqueness only; the lock provides the mutual exclusion.)
fn fill_nonce() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", std::process::id(), nanos)
}

/// An exclusive advisory lock held for the duration of one cache fill, released
/// when the file handle closes. Mirrors the pack store's `flock` discipline so
/// two fillers of the same key never interleave — including two async tasks in
/// ONE process, which a pid-keyed staging path alone does not separate.
struct FillLock(
    // Never read: the handle IS the lock. The OS releases the advisory lock when
    // this file closes, so the field's only job is to live as long as the guard.
    #[allow(dead_code)] std::fs::File,
);

impl FillLock {
    fn acquire(path: &Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| Error::config("image-store: lock", e.to_string()))?;
        lock_exclusive(&file).map_err(|e| Error::config("image-store: lock", e.to_string()))?;
        Ok(Self(file))
    }
}

#[cfg(unix)]
fn lock_exclusive(file: &std::fs::File) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // SAFETY: `fd` is a valid open descriptor for the lifetime of the call.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
fn lock_exclusive(file: &std::fs::File) -> std::io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{LockFileEx, LOCKFILE_EXCLUSIVE_LOCK};
    use windows_sys::Win32::System::IO::OVERLAPPED;
    let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
    // SAFETY: handle is valid; overlapped is a zeroed, correctly sized struct.
    let ok = unsafe {
        LockFileEx(
            file.as_raw_handle() as _,
            LOCKFILE_EXCLUSIVE_LOCK,
            0,
            !0,
            !0,
            &mut overlapped,
        )
    };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Remove staging debris left by a crashed fill. Without this every crash leaks
/// a full extracted image (hundreds of MiB) that nothing ever reclaims.
fn sweep_staging(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(".staging-") {
            continue;
        }
        // Only reap debris that is demonstrably stale, so a concurrent fill on
        // another node process is never pulled out from under itself.
        let stale = e
            .metadata()
            .and_then(|m| m.modified())
            .map(|t| {
                t.elapsed()
                    .map(|age| age > std::time::Duration::from_secs(3600))
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        if stale {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// Evict least-recently-used entries until the store fits `max_bytes`, never
/// removing `keep` (the entry the caller is about to use).
fn prune_store(root: &Path, max_bytes: u64, keep: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut items: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
    let mut total = 0u64;
    for e in entries.flatten() {
        let path = e.path();
        let name = e.file_name();
        let name = name.to_string_lossy();
        // Only content entries participate; locks and staging are not evictable.
        if name.starts_with('.') || !path.is_dir() {
            continue;
        }
        let size = dir_size(&path);
        total += size;
        let mtime = e
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        items.push((mtime, size, path));
    }
    if total <= max_bytes {
        return;
    }
    // Oldest first.
    items.sort_by_key(|(mtime, _, _)| *mtime);
    for (_, size, path) in items {
        if total <= max_bytes {
            break;
        }
        if path == keep {
            continue;
        }
        if std::fs::remove_dir_all(&path).is_ok() {
            total = total.saturating_sub(size);
        }
    }
}

/// Recursive on-disk size of a directory, following no symlinks.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for e in entries.flatten() {
        let Ok(md) = e.metadata() else { continue };
        if md.is_dir() {
            total += dir_size(&e.path());
        } else {
            total += md.len();
        }
    }
    total
}

/// A cache entry is usable only if its cached `config.json` is present AND its
/// `layers/` dir holds the layer-order index. The layer-order index is written
/// last, so its presence means the whole entry (layers + config) is complete —
/// a crash mid-fill never yields a partial cache hit.
fn is_intact(entry: &Path) -> bool {
    entry.join(CONFIG_FILE).is_file() && entry.join("layers").join("layer-order").is_file()
}

/// Materialize a full cache entry into `staging`: EXTRACT each layer into
/// `staging/layers/<digest>/` as an overlayfs-ready lowerdir (decompressed,
/// untarred, OCI whiteouts translated to overlay char-devices — the same
/// `smolvm_oci_layer::extract_oci_layer` the guest agent uses), cache the image
/// config as `staging/config.json`, and write the layer-order index LAST so the
/// entry only looks intact once everything is present. Returns the parsed config.
/// The guest then mounts the layer dirs read-only as overlay lowerdirs (via the
/// packed-layers path) and boots the config's command with no in-VM pull/flatten.
///
/// Extraction is Linux+root: whiteout translation needs `mknod`/`setxattr`, and
/// ownership preservation needs `CAP_CHOWN`. Both hold on the node (root); the
/// host image-cache path is not used on macOS (which keeps the bake path), so
/// the non-Linux `extract_oci_layer` stub returning `Unsupported` is never hit
/// in a real cache-fill.
async fn materialize_entry(
    client: &smolvm_registry::RegistryClient,
    repo: &str,
    manifest: &smolvm_registry::OciManifest,
    staging: &Path,
) -> Result<ImageConfig> {
    let layers_dir = staging.join("layers");
    std::fs::create_dir_all(&layers_dir)
        .map_err(|e| Error::config("image-store", e.to_string()))?;

    let mut order = Vec::with_capacity(manifest.layers.len());
    for descriptor in &manifest.layers {
        smolvm_registry::validate_digest(&descriptor.digest)
            .map_err(|e| Error::agent("image-store: layer digest", e.to_string()))?;
        // Stream the blob to disk rather than `pull_blob`, which buffers the whole
        // layer in memory and caps it at 64 MiB — a limit essentially every real
        // base image (python, node, cuda) exceeds. The digest is verified from the
        // streamed bytes before extraction, so a mirror or peer still cannot
        // substitute different content under the same address.
        let blob_path = staging.join(format!("{}.blob", digest_dir(&descriptor.digest)));
        stream_blob_verified(client, repo, &descriptor.digest, &blob_path).await?;

        let dir = layers_dir.join(digest_dir(&descriptor.digest));
        std::fs::create_dir_all(&dir).map_err(|e| Error::config("image-store", e.to_string()))?;
        let blob = std::fs::File::open(&blob_path)
            .map_err(|e| Error::config("image-store: open layer", e.to_string()))?;
        let extracted = smolvm_oci_layer::extract_oci_layer(std::io::BufReader::new(blob), &dir)
            .map_err(|e| Error::agent("image-store: extract layer", e.to_string()));
        // The compressed blob is scratch — drop it either way so a fill never
        // leaves both representations on disk.
        let _ = std::fs::remove_file(&blob_path);
        extracted?;
        order.push(digest_dir(&descriptor.digest));
    }

    // Fetch + cache the image config (entrypoint/cmd/env/workdir). pull_blob
    // verifies the blob against the config descriptor digest before we parse it.
    smolvm_registry::validate_digest(&manifest.config.digest)
        .map_err(|e| Error::agent("image-store: config digest", e.to_string()))?;
    let config_bytes = client
        .pull_blob(repo, &manifest.config.digest)
        .await
        .map_err(|e| Error::agent("image-store: pull config", e.to_string()))?;
    let config: ImageConfig = serde_json::from_slice::<OciConfigBlob>(&config_bytes)
        .map_err(|e| Error::agent("image-store: parse config", e.to_string()))?
        .into();
    let config_json = serde_json::to_vec(&config)
        .map_err(|e| Error::agent("image-store: encode config", e.to_string()))?;
    std::fs::write(staging.join(CONFIG_FILE), &config_json)
        .map_err(|e| Error::config("image-store", e.to_string()))?;

    // Written LAST: its presence is the "entry is complete" marker `is_intact`
    // checks (config.json is already on disk above), so a crash never hits.
    std::fs::write(layers_dir.join("layer-order"), order.join("\n"))
        .map_err(|e| Error::config("image-store", e.to_string()))?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_dir_is_filesystem_safe() {
        assert_eq!(digest_dir("sha256:abc123"), "sha256-abc123");
    }

    /// THE key-scoping property: the same manifest digest served from a DIFFERENT
    /// registry (or repo) must land on a different entry. Without this, an
    /// attacker holding a private image's manifest bytes could serve them from a
    /// registry they control, pass the gate there, and be handed the cached
    /// private layers.
    #[test]
    fn entry_key_is_scoped_to_registry_and_repo() {
        let d = "sha256:abc";
        let real = entry_key("registry.example.com", "team/private", d);
        assert_ne!(
            real,
            entry_key("evil.example.com", "team/private", d),
            "same digest from another REGISTRY must not share an entry"
        );
        assert_ne!(
            real,
            entry_key("registry.example.com", "other/repo", d),
            "same digest under another REPO must not share an entry"
        );
        assert_ne!(
            real,
            entry_key("registry.example.com", "team/private", "sha256:def"),
            "a different digest is a different entry"
        );
        // Dedup still holds for the identical (registry, repo, digest) triple.
        assert_eq!(real, entry_key("registry.example.com", "team/private", d));
        // And the key is a filesystem-safe single component.
        assert!(!real.contains('/') && !real.contains(':'));
    }

    /// The store must stay bounded: extracted layers are far larger than the
    /// blobs they came from, so an unbounded store fills a node's disk and every
    /// tenant on it fails to start. Eviction is LRU and never drops `keep`.
    #[test]
    fn prune_store_evicts_oldest_and_never_the_kept_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let mk = |name: &str, bytes: usize, age_secs: u64| -> PathBuf {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("blob"), vec![0u8; bytes]).unwrap();
            let f = std::fs::File::open(&dir).unwrap();
            let _ = f.set_modified(
                std::time::SystemTime::now() - std::time::Duration::from_secs(age_secs),
            );
            dir
        };
        let oldest = mk("a", 4096, 300);
        let middle = mk("b", 4096, 200);
        let newest = mk("c", 4096, 100);
        // Cap admits two of the three entries → exactly one eviction.
        prune_store(root, 9000, &newest);
        assert!(!oldest.exists(), "least-recently-used entry evicted");
        assert!(middle.exists(), "entry within the cap survives");
        assert!(newest.exists(), "the kept entry is never evicted");
    }

    /// A `keep` that is itself the oldest must survive even when over the cap —
    /// evicting the entry the caller is about to boot would be self-defeating.
    #[test]
    fn prune_store_keeps_the_active_entry_even_when_oldest() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let old = root.join("old");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("blob"), vec![0u8; 8192]).unwrap();
        let f = std::fs::File::open(&old).unwrap();
        let _ = f.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(999));
        prune_store(root, 1, &old);
        assert!(old.exists(), "the active entry survives an over-cap prune");
    }

    #[test]
    fn intact_requires_config_and_order_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let entry = tmp.path().join("sha256-x");
        std::fs::create_dir_all(entry.join("layers").join("sha256-l1")).unwrap();
        assert!(
            !is_intact(&entry),
            "no config or order marker → not a valid hit"
        );
        // The config alone is not enough — layers may still be materializing.
        std::fs::write(entry.join(CONFIG_FILE), b"{}").unwrap();
        assert!(
            !is_intact(&entry),
            "config but no order marker → not a valid hit"
        );
        // The order marker is written last; with the config already present, its
        // arrival means the whole entry is complete.
        std::fs::write(entry.join("layers").join("layer-order"), "sha256-l1").unwrap();
        assert!(is_intact(&entry), "config + order marker → valid hit");
    }

    #[test]
    fn repo_for_maps_docker_hub_and_namespaced_refs() {
        // Docker Hub official image (no namespace) → implicit `library/`.
        let bare = Reference::parse("alpine").unwrap();
        assert_eq!(repo_for("docker.io", &bare), "library/alpine");
        // A namespaced ref keeps its namespace on any host.
        let ns = Reference::parse("ghcr.io/org/tool:v1").unwrap();
        assert_eq!(repo_for("ghcr.io", &ns), "org/tool");
        // A bare name on a non-Docker-Hub host is not `library/`-prefixed.
        assert_eq!(repo_for("ghcr.io", &bare), "alpine");
    }

    /// THE security property: a cache HIT must not bypass authorization. Even
    /// with the layers already on disk, an unauthorized caller is rejected at the
    /// gate; only an authorized caller is served the cached entry.
    #[test]
    fn cache_hit_still_requires_authorization() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let server = MockServer::start().await;
            let body = br#"{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json","config":{"mediaType":"application/vnd.oci.image.config.v1+json","digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000","size":0},"layers":[]}"#.to_vec();

            // Authorized bearer → 200 with the manifest.
            Mock::given(method("GET"))
                .and(path("/v2/myrepo/manifests/latest"))
                .and(header("authorization", "Bearer good-token"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .insert_header("content-type", "application/vnd.oci.image.manifest.v1+json")
                        .set_body_bytes(body.clone()),
                )
                .with_priority(1)
                .mount(&server)
                .await;
            // Anyone else → 401 (no challenge header → the client does not retry).
            Mock::given(method("GET"))
                .and(path("/v2/myrepo/manifests/latest"))
                .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
                .with_priority(5)
                .mount(&server)
                .await;

            let host = server.uri().strip_prefix("http://").unwrap().to_string();
            let reference = format!("{host}/myrepo:latest");

            // Pre-create a VALID cache entry under the SCOPED key (registry +
            // repo + manifest digest) — the bytes are already present on disk.
            let digest = format!("sha256:{}", hex::encode(Sha256::digest(&body)));
            let tmp = tempfile::tempdir().unwrap();
            let store = ImageStore::new(tmp.path().to_path_buf());
            let entry = tmp.path().join(entry_key(&host, "myrepo", &digest));
            let layers = entry.join("layers");
            std::fs::create_dir_all(&layers).unwrap();
            std::fs::write(layers.join("layer-order"), "").unwrap();
            // An intact entry also carries the cached image config.
            std::fs::write(entry.join(CONFIG_FILE), b"{\"entrypoint\":[\"/bin/true\"],\"cmd\":[],\"env\":[],\"workdir\":\"\"}").unwrap();
            assert!(is_intact(&entry));

            // DENY: unauthorized caller is rejected at the gate despite the hit.
            let denied = store.ensure_image(&reference, &PullAuth::Anonymous).await;
            assert!(denied.is_err(), "cache hit must NOT bypass authorization");

            // ALLOW: authorized caller resolves and is served the cached entry,
            // including the config parsed back from disk.
            let allowed = store
                .ensure_image(&reference, &PullAuth::Bearer("good-token".into()))
                .await
                .expect("authorized caller should be served");
            assert_eq!(allowed.layers, layers);
            assert_eq!(allowed.config.entrypoint, vec!["/bin/true".to_string()]);
        });
    }
}
