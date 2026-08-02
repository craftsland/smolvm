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
    pub async fn ensure_image(&self, reference: &str, auth: &PullAuth) -> Result<PathBuf> {
        // ── AUTH GATE + content address ─────────────────────────────────────
        // Resolve + authorize the manifest with the caller's credentials, across
        // the candidate registries the guest would try. Runs before the cache is
        // consulted, so a hit cannot leak an image the caller can't pull.
        let r = resolve_authorized(reference, auth).await?;
        let entry = self.root.join(digest_dir(&r.digest));
        let layers = entry.join("layers");

        // ── CACHE HIT (already authorized above) ───────────────────────────
        if is_intact(&entry) {
            return Ok(layers);
        }

        // ── MISS: materialize host-side, verify digests, atomic-stage ──────
        let manifest: smolvm_registry::OciManifest = serde_json::from_slice(&r.manifest_bytes)
            .map_err(|e| Error::agent("image-store: parse manifest", e.to_string()))?;
        let staging = self.root.join(format!(
            ".staging-{}-{}",
            digest_dir(&r.digest),
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&staging);
        materialize_layers(&r.client, &r.repo, &manifest, &staging)
            .await
            .inspect_err(|_| {
                let _ = std::fs::remove_dir_all(&staging);
            })?;

        // Atomic publish: the last writer wins; a crash leaves a `.staging-*`
        // dir the next run overwrites, never a half-populated cache entry.
        if let Some(parent) = entry.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::config("image-store", e.to_string()))?;
        }
        match std::fs::rename(&staging, &entry) {
            Ok(()) => {}
            Err(_) if is_intact(&entry) => {
                // A concurrent caller published the same digest first — fine.
                let _ = std::fs::remove_dir_all(&staging);
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(Error::config("image-store: publish", e.to_string()));
            }
        }
        Ok(layers)
    }
}

/// The outcome of the auth gate: the authorized client + repo it succeeded
/// against, the manifest bytes, and their digest (the content-address key).
struct Resolved {
    client: smolvm_registry::RegistryClient,
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
        .map_err(|e| Error::config("image-store", format!("bad reference: {}", e.reason)))?;
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
                    repo,
                    manifest_bytes,
                    digest,
                });
            }
            Err(e) => last_err = Some(e.to_string()),
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

/// A cache entry is usable only if its `layers/` dir holds the layer-order index
/// (written last), guarding against a partial extraction being treated as valid.
fn is_intact(entry: &Path) -> bool {
    entry.join("layers").join("layer-order").is_file()
}

/// Pull each layer blob, verify it against its digest, write it under
/// `staging/layers/<digest>/`, and record the stacking order last so the entry
/// only looks intact once every layer is present.
async fn materialize_layers(
    client: &smolvm_registry::RegistryClient,
    repo: &str,
    manifest: &smolvm_registry::OciManifest,
    staging: &Path,
) -> Result<()> {
    let layers_dir = staging.join("layers");
    std::fs::create_dir_all(&layers_dir)
        .map_err(|e| Error::config("image-store", e.to_string()))?;

    let mut order = Vec::with_capacity(manifest.layers.len());
    for descriptor in &manifest.layers {
        smolvm_registry::validate_digest(&descriptor.digest)
            .map_err(|e| Error::agent("image-store: layer digest", e.to_string()))?;
        // pull_blob verifies the returned bytes hash to `digest`, so a mirror or
        // peer cannot substitute different content under the same address.
        let blob = client
            .pull_blob(repo, &descriptor.digest)
            .await
            .map_err(|e| Error::agent("image-store: pull layer", e.to_string()))?;
        let dir = layers_dir.join(digest_dir(&descriptor.digest));
        std::fs::create_dir_all(&dir).map_err(|e| Error::config("image-store", e.to_string()))?;
        std::fs::write(dir.join("layer.tar"), &blob)
            .map_err(|e| Error::config("image-store", e.to_string()))?;
        order.push(digest_dir(&descriptor.digest));
    }
    // Written last: its presence is the "entry is complete" marker `is_intact`
    // checks, so a crash mid-pull never yields a cache hit.
    std::fs::write(layers_dir.join("layer-order"), order.join("\n"))
        .map_err(|e| Error::config("image-store", e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_dir_is_filesystem_safe() {
        assert_eq!(digest_dir("sha256:abc123"), "sha256-abc123");
    }

    #[test]
    fn intact_requires_the_order_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let entry = tmp.path().join("sha256-x");
        std::fs::create_dir_all(entry.join("layers").join("sha256-l1")).unwrap();
        assert!(!is_intact(&entry), "no order marker yet → not a valid hit");
        std::fs::write(entry.join("layers").join("layer-order"), "sha256-l1").unwrap();
        assert!(is_intact(&entry), "order marker present → valid hit");
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

            // Pre-create a VALID cache entry for the manifest digest — the bytes
            // are already present on disk.
            let digest = format!("sha256:{}", hex::encode(Sha256::digest(&body)));
            let tmp = tempfile::tempdir().unwrap();
            let store = ImageStore::new(tmp.path().to_path_buf());
            let layers = tmp.path().join(digest_dir(&digest)).join("layers");
            std::fs::create_dir_all(&layers).unwrap();
            std::fs::write(layers.join("layer-order"), "").unwrap();
            assert!(is_intact(&tmp.path().join(digest_dir(&digest))));

            // DENY: unauthorized caller is rejected at the gate despite the hit.
            let denied = store.ensure_image(&reference, &PullAuth::Anonymous).await;
            assert!(denied.is_err(), "cache hit must NOT bypass authorization");

            // ALLOW: authorized caller resolves and is served the cached layers.
            let allowed = store
                .ensure_image(&reference, &PullAuth::Bearer("good-token".into()))
                .await;
            assert_eq!(
                allowed.expect("authorized caller should be served"),
                layers
            );
        });
    }
}
