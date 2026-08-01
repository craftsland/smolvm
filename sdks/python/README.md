# smolvm rollout client

`smolvm-rollout` is the framework-neutral generation boundary for fused policy
rollouts. A trainer publishes each immutable LoRA version and submits text or
token-ID prompts; smolvm verifies and routes the version while one local vLLM
engine continuously batches compatible policies. Unsupported workflows can use
the executor's advertised `fallbackPool` with the ordinary fork-lease API.

The vLLM server must bind to loopback, enable LoRA and runtime adapter updates,
and reserve at least one spare CPU LoRA slot so a new policy version can load
before the previous version retires. smolvm never exposes vLLM's unrestricted
adapter-path endpoint to remote callers.

```python
from smolvm_rollout import RolloutClient

client = RolloutClient("http://127.0.0.1:8080/api/v1", "qwen")
client.ensure_vllm_executor(
    endpoint="http://127.0.0.1:8000",
    adapter_root="/var/lib/smolvm/adapters",
    fallback_pool="isolated-rollouts",
)
client.publish_policy("experiment-a", "step-40", "/var/lib/smolvm/adapters/a-40")
result = client.generate(
    idempotency_key="experiment-a-step-40-batch-7",
    policy="experiment-a",
    prompts=[[1, 2, 3]],
    max_tokens=64,
    temperature=0.9,
    seed=7,
    logprobs=1,
)
```
