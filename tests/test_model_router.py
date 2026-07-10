import json
import unittest

from ops.model_router_core import parse_backends, select_model


class ModelRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backends = {
            "SubTokenLLM": "http://127.0.0.1:8888/v1",
            "SubTokenLLM-E2B": "http://127.0.0.1:8889/v1",
        }

    def test_selects_e2b_without_mutating_request_bytes(self) -> None:
        body = json.dumps(
            {
                "model": "SubTokenLLM-E2B",
                "messages": [{"role": "user", "content": "hello"}],
            },
            separators=(",", ":"),
        ).encode()
        original = bytes(body)
        self.assertEqual(select_model(body, self.backends, "SubTokenLLM"), "SubTokenLLM-E2B")
        self.assertEqual(body, original)

    def test_defaults_to_e4b_and_rejects_unknown_model(self) -> None:
        self.assertEqual(select_model(b"", self.backends, "SubTokenLLM"), "SubTokenLLM")
        with self.assertRaises(ValueError):
            select_model(b'{"model":"unknown"}', self.backends, "SubTokenLLM")

    def test_parses_backend_json(self) -> None:
        self.assertEqual(
            parse_backends('{"m":"http://localhost:1/v1/"}'),
            {"m": "http://localhost:1/v1"},
        )


if __name__ == "__main__":
    unittest.main()
