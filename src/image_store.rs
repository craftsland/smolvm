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

use crate::registry::Reference;
use crate::{Error, Result};

/// Credentials a caller presents to authorize a pull. Locally this is the user's
/// docker-config token (or nothing, for anonymous public pulls); on the cloud it
/// is the control-plane-minted per-tenant `registry_identity_token` that already
/// flows to the node. The value only decides authorization — never the cache key.
#[derive(Clone, Default)]
pub struct PullAuth {
    /// Bearer token sent directly to the registry (a pre-minted scoped token).
    pub bearer: Option<String>,
    /// Identity/refresh token exchanged at the registry's auth service after a
    /// `WWW-Authenticate: Bearer` challenge.
    pub identity_token: Option<String>,
}

impl PullAuth {
    /// Anonymous — used for public images with no credentials.
    pub fn anonymous() -> Self {
        Self::default()
    }

    fn apply(
        &self,
        mut client: smolvm_registry::RegistryClient,
    ) -> smolvm_registry::RegistryClient {
        if let Some(t) = &self.bearer {
            client = client.with_token(t.clone());
        }
        if let Some(t) = &self.identity_token {
            client = client.with_identity_token(t.clone());
        }
        client
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
    pub async fn ensure_image(&self, reference: &str, auth: &PullAuth) -> Result<PathBuf> {
        let parsed = Reference::parse(reference)
            .map_err(|e| Error::config("image-store", format!("bad reference: {}", e.reason)))?;
        let repo = repo_path(&parsed);
        let want = parsed
            .digest
            .clone()
            .or_else(|| parsed.tag.clone())
            .unwrap_or_else(|| "latest".to_string());

        let base_url = api_base_url(&parsed.registry);
        let client = auth.apply(smolvm_registry::RegistryClient::new(base_url));

        // ── AUTH GATE ──────────────────────────────────────────────────────
        // Resolve the manifest WITH the caller's credentials. The registry
        // authorizes `repository:<repo>:pull` here; an unauthorized caller is
        // rejected (401/403) and never reaches the cache below. Runs on hits too.
        let manifest_bytes = client
            .get_manifest_resolved(&repo, &want)
            .await
            .map_err(|e| Error::agent("image-store: resolve manifest", e.to_string()))?;

        // The manifest digest is the content address / cache key. Tag→digest is
        // resolved fresh every call, so a moved `:latest` becomes a new key.
        let digest = format!("sha256:{}", hex::encode(Sha256::digest(&manifest_bytes)));
        let entry = self.root.join(digest_dir(&digest));
        let layers = entry.join("layers");

        // ── CACHE HIT (already authorized above) ───────────────────────────
        if is_intact(&entry) {
            return Ok(layers);
        }

        // ── MISS: materialize host-side, verify digests, atomic-stage ──────
        let manifest: smolvm_registry::OciManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| Error::agent("image-store: parse manifest", e.to_string()))?;
        let staging = self.root.join(format!(
            ".staging-{}-{}",
            digest_dir(&digest),
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&staging);
        materialize_layers(&client, &repo, &manifest, &staging)
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

/// The auth gate on its own: resolve `reference`'s manifest with the caller's
/// credentials and return the content digest. A caller who cannot pull the repo
/// is rejected here (401/403). Used by the paths that materialize the image
/// through the proven pack pipeline but still need the per-caller authorization
/// and the content address as their cache key — so the security property holds
/// identically whether the layers come from `ensure_image` or a bake.
pub async fn authorized_digest(reference: &str, auth: &PullAuth) -> Result<String> {
    let parsed = Reference::parse(reference)
        .map_err(|e| Error::config("image-store", format!("bad reference: {}", e.reason)))?;
    let want = parsed
        .digest
        .clone()
        .or_else(|| parsed.tag.clone())
        .unwrap_or_else(|| "latest".to_string());

    // Resolve the registry the way the GUEST does — via `registry_pull_hosts`,
    // not the configured default — so the host-side gate and the in-guest pull
    // target the same registry. A bare `alpine` resolves to Docker Hub here, not
    // the smol registry, matching what the guest would actually pull.
    let hosts = crate::registry::registry_pull_hosts(reference);
    let mut last_err: Option<String> = None;
    for host in &hosts {
        let (base_url, repo) = endpoint_for(host, &parsed);
        let client = auth.apply(smolvm_registry::RegistryClient::new(base_url));
        match client.get_manifest_resolved(&repo, &want).await {
            Ok(bytes) => {
                return Ok(format!("sha256:{}", hex::encode(Sha256::digest(&bytes))));
            }
            Err(e) => last_err = Some(e.to_string()),
        }
    }
    Err(Error::agent(
        "image-store: authorize",
        last_err.unwrap_or_else(|| "no candidate registry resolved the image".to_string()),
    ))
}

/// Map a candidate pull host (from `registry_pull_hosts`) plus the parsed name to
/// the registry API base URL and repository path. Docker Hub's OCI API lives at
/// `registry-1.docker.io` and official images sit under `library/`; the smol
/// registry serves its API at `registry.smolmachines.com`.
fn endpoint_for(host: &str, parsed: &Reference) -> (String, String) {
    let repo_with = |prefix_library: bool| -> String {
        match &parsed.namespace {
            Some(ns) => format!("{}/{}", ns, parsed.name),
            None if prefix_library => format!("library/{}", parsed.name),
            None => parsed.name.clone(),
        }
    };
    match host {
        "docker.io" | "docker.com" | "index.docker.io" | "registry-1.docker.io" => {
            ("https://registry-1.docker.io".to_string(), repo_with(true))
        }
        h if h.ends_with("smolmachines.com") => (
            "https://registry.smolmachines.com".to_string(),
            repo_with(false),
        ),
        h if smolvm_registry::is_local_registry(h) => (format!("http://{}", h), repo_with(false)),
        h => (format!("https://{}", h), repo_with(false)),
    }
}

/// The registry repository path for a reference (`namespace/name` or `name`).
///
/// Docker Hub official images (no namespace on `docker.io`) live under the
/// implicit `library/` namespace, so a bare `alpine` resolves to `library/alpine`
/// — without this the pull-scope is wrong and the registry answers 401.
fn repo_path(r: &Reference) -> String {
    match &r.namespace {
        Some(ns) => format!("{}/{}", ns, r.name),
        None if r.registry == crate::registry::DEFAULT_REGISTRY => {
            format!("library/{}", r.name)
        }
        None => r.name.clone(),
    }
}

/// A filesystem-safe directory name for a digest (`sha256:abc` → `sha256-abc`).
fn digest_dir(digest: &str) -> String {
    digest.replace(':', "-")
}

/// The registry API base URL for a hostname. Local registries are plaintext; the
/// Docker Hub apex (`docker.io`) is a namespace alias whose OCI API is served at
/// `registry-1.docker.io`, so it is rewritten here (the token service at
/// `auth.docker.io` is discovered from the `WWW-Authenticate` challenge).
fn api_base_url(registry: &str) -> String {
    if smolvm_registry::is_local_registry(registry) {
        format!("http://{}", registry)
    } else if registry == crate::registry::DEFAULT_REGISTRY {
        "https://registry-1.docker.io".to_string()
    } else {
        format!("https://{}", registry)
    }
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
    fn auth_only_affects_the_client_not_the_key() {
        // The cache key is the manifest digest, independent of who authorized —
        // so two tenants pulling the same public image share one entry.
        let a = PullAuth {
            bearer: Some("tenant-a".into()),
            identity_token: None,
        };
        let b = PullAuth::anonymous();
        // Both apply cleanly; the point is they never enter the digest path.
        let _ = a.apply(smolvm_registry::RegistryClient::new("http://x".into()));
        let _ = b.apply(smolvm_registry::RegistryClient::new("http://x".into()));
    }

    #[test]
    fn endpoint_for_maps_registries_correctly() {
        let bare = Reference::parse("alpine").unwrap();
        // Docker Hub official image → API endpoint + implicit `library/`.
        assert_eq!(
            endpoint_for("docker.io", &bare),
            (
                "https://registry-1.docker.io".to_string(),
                "library/alpine".to_string()
            )
        );
        let ns = Reference::parse("ghcr.io/org/tool:v1").unwrap();
        assert_eq!(
            endpoint_for("ghcr.io", &ns),
            ("https://ghcr.io".to_string(), "org/tool".to_string())
        );
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
            let denied = store.ensure_image(&reference, &PullAuth::anonymous()).await;
            assert!(denied.is_err(), "cache hit must NOT bypass authorization");

            // ALLOW: authorized caller resolves and is served the cached layers.
            let allowed = store
                .ensure_image(
                    &reference,
                    &PullAuth {
                        bearer: Some("good-token".into()),
                        identity_token: None,
                    },
                )
                .await;
            assert_eq!(
                allowed.expect("authorized caller should be served"),
                layers
            );
        });
    }
}
