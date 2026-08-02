import tempfile
import unittest
from pathlib import Path

from smolvm_rollout import RolloutClient, RolloutError, adapter_sha256


class AdapterDigestTests(unittest.TestCase):
    def test_digest_is_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_bytes(b"{}")
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            first = adapter_sha256(root)
            self.assertEqual(
                first,
                "26d1c7593b9650cb489a9a1fe2fad9def32c75ec2685cf8261c3c0fa3b73e315",
            )
            self.assertEqual(first, adapter_sha256(root))
            (root / "adapter_model.safetensors").write_bytes(b"changed")
            self.assertNotEqual(first, adapter_sha256(root))


class RecordingClient(RolloutClient):
    def __init__(self):
        super().__init__("http://127.0.0.1:1/api/v1", "fused")
        self.recorded = None

    def _request(self, method, path, body=None):
        self.recorded = (method, path, body)
        return {"ok": True}


class DevicePublicationTests(unittest.TestCase):
    def test_device_token_is_encoded_without_exposing_a_descriptor(self):
        client = RecordingClient()
        response = client.publish_device_policy(
            "policy", "step-1", bytes(range(32))
        )
        self.assertEqual(response, {"ok": True})
        method, path, body = client.recorded
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/rollout-executors/fused/device-policies")
        self.assertEqual(body["tensorBundleToken"], bytes(range(32)).hex())
        self.assertNotIn("adapterSha256", body)

    def test_device_publication_retries_one_ambiguous_transport_failure(self):
        class FlakyClient(RecordingClient):
            def __init__(self):
                super().__init__()
                self.attempts = []

            def _request(self, method, path, body=None):
                self.attempts.append((method, path, body))
                if len(self.attempts) == 1:
                    raise RolloutError(0, "UNAVAILABLE", "lost response")
                return {"ok": True}

        client = FlakyClient()
        self.assertEqual(
            client.publish_device_policy("policy", "step-1", bytes(range(32))),
            {"ok": True},
        )
        self.assertEqual(len(client.attempts), 2)
        self.assertEqual(client.attempts[0], client.attempts[1])


if __name__ == "__main__":
    unittest.main()
