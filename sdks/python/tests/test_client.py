import tempfile
import unittest
from pathlib import Path

from smolvm_rollout import adapter_sha256


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


if __name__ == "__main__":
    unittest.main()
