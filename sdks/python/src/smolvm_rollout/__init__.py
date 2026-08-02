"""Framework-neutral client for smolvm fused rollout executors."""

from .client import RolloutClient, RolloutError, adapter_sha256

__all__ = ["RolloutClient", "RolloutError", "adapter_sha256"]
