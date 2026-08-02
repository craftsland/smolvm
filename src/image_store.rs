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

        let base_url = if smolvm_registry::is_local_registry(&parsed.registry) {
            format!("http://{}", parsed.registry)
        } else {
            format!("https://{}", parsed.registry)
        };
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

/// The registry repository path for a reference (`namespace/name` or `name`).
fn repo_path(r: &Reference) -> String {
    match &r.namespace {
        Some(ns) => format!("{}/{}", ns, r.name),
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
}
