import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from frozen_inference.backend import BackendError, ChatClient, merge_usage
from frozen_inference.cli import main, read_jsonl, select_examples, summary
from frozen_inference.executor import Rejected, digest_text
from frozen_inference.memory import SkillLibrary
from frozen_inference.pipeline import Config, Pipeline, json_object, render_answer
from frozen_inference.policy import RepairPolicy, features, fit_policy

ROOT = Path(__file__).resolve().parents[1]


def call_record(content, reason="stop"):
    return {"content": content, "finish_reason": reason,
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                      "completion_tokens_details": {"reasoning_tokens": 2}},
            "usage_complete": True, "attempts": [{"elapsed_seconds": 0.01, "error": None}]}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, dict) else call_record(value)


def tracking_ir(irrelevant=False):
    return {"kind": "tracking", "complete": True,
            "initial": {"a": "red", "b": "blue", "c": "green", "d": "black"}, "query": "a",
            "steps": [{"source": "swap", "alternatives": [["b", "c"], ["c", "d"]] if irrelevant else
                       [["a", "b"], ["a", "c"]]}]}


class PipelineTests(unittest.TestCase):
    def test_zero_call_is_real(self):
        client = FakeClient([])
        result = Pipeline(client).predict("not False or False is")
        self.assertEqual((result["prediction"], result["model_calls"], result["route"]), ("True", 0, "exact"))
        self.assertEqual(client.prompts, [])
        self.assertTrue(result["usage_complete"])

    def test_compilation_requires_opt_in(self):
        client = FakeClient(["answer"])
        result = Pipeline(client).predict("Read the swap story.")
        self.assertEqual(result["route"], "fallback")
        self.assertEqual(result["calls"][0]["action"], "direct")

    def test_stable_skips_irrelevant_repair(self):
        client = FakeClient([json.dumps(tracking_ir(True))])
        result = Pipeline(client, Config(enable_compile=True)).predict("Read the swap story.")
        self.assertEqual(result["prediction"], "red")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["certificate_scope"], "conditional-on-model-interpretation")

    def test_full_mode_resolves_even_irrelevant_ambiguity(self):
        client = FakeClient([json.dumps(tracking_ir(True)), '{"choice":0}'])
        result = Pipeline(client, Config(enable_compile=True, mode="full")).predict("Read the swap story.")
        self.assertEqual(result["prediction"], "red")
        self.assertEqual(result["model_calls"], 2)

    def test_answer_changing_repair_and_mapping(self):
        client = FakeClient([json.dumps(tracking_ir()), '{"choice":1}'])
        result = Pipeline(client, Config(enable_compile=True)).predict("Read the swap story.\n(A) blue\n(B) green")
        self.assertEqual(result["prediction"], "(B)")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["events"], [{"repair_step": 0, "choice": 1}])

    def test_invalid_json_falls_back_and_counts_both_calls(self):
        client = FakeClient(["not JSON", "B"])
        result = Pipeline(client, Config(enable_compile=True)).predict("Read the swap story.")
        self.assertEqual(result["prediction"], "B")
        self.assertEqual(result["route"], "fallback")
        self.assertEqual(result["usage"]["completion_tokens"], 14)
        self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 4)

    def test_truncated_compiler_is_not_accepted(self):
        client = FakeClient([call_record(json.dumps(tracking_ir(True)), "length"), "fallback"])
        result = Pipeline(client, Config(enable_compile=True)).predict("Read the swap story.")
        self.assertEqual(result["prediction"], "fallback")
        self.assertEqual(result["model_calls"], 2)

    def test_reserves_fallback_call(self):
        client = FakeClient([json.dumps(tracking_ir()), "fallback"])
        result = Pipeline(client, Config(enable_compile=True, max_calls=2)).predict("Read the swap story.")
        self.assertEqual([c["action"] for c in result["calls"]], ["compile", "direct"])
        self.assertEqual(result["prediction"], "fallback")

    def test_null_or_bool_repair_is_not_a_choice(self):
        for response in ['{"choice":null}', '{"choice":true}', '{"choice":0,"extra":1}']:
            client = FakeClient([json.dumps(tracking_ir()), response, "fallback"])
            result = Pipeline(client, Config(enable_compile=True)).predict("Read the swap story.")
            self.assertEqual(result["prediction"], "fallback")
            self.assertEqual(result["model_calls"], 3)

    def test_failed_request_cost_is_retained(self):
        failed = {"attempts": [{"error": "timeout", "elapsed_seconds": 1.0}], "usage_complete": False}
        client = FakeClient([BackendError("oops", failed), "fallback"])
        result = Pipeline(client, Config(enable_compile=True)).predict("Read the swap story.")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["request_attempts"], 2)
        self.assertFalse(result["usage_complete"])

    def test_no_backend_never_fabricates_answer(self):
        result = Pipeline().predict("Unknown natural language question")
        self.assertEqual(result["prediction"], "")
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["route"], "unanswered")

    def test_direct_control_bypasses_all_fast_paths(self):
        client = FakeClient(["model output"])
        result = Pipeline(client, Config(mode="direct")).predict("2+3")
        self.assertEqual(result["prediction"], "model output")
        self.assertEqual(result["route"], "direct")

    def test_seed_reproducibility(self):
        a, b = FakeClient(["a"]), FakeClient(["a"])
        Pipeline(a).predict("Question")
        Pipeline(b).predict("Question")
        self.assertEqual(a.prompts[0][1]["seed"], b.prompts[0][1]["seed"])

    def test_nonfinal_or_duplicate_option_mapping_rejects(self):
        for question in ["q\n(A) red\n(B) red", "q\n(A) red\n(B) blue\nExplain."]:
            with self.assertRaises(Rejected):
                render_answer(question, "red")
        self.assertEqual(render_answer("q\nA) red\nB) blue", "blue"), "(B)")

    def test_duplicate_nonfinite_or_list_json_rejected(self):
        for text in ['{"x":1,"x":2}', '{"x":NaN}', '[1,2]', '{"x":Infinity}']:
            with self.assertRaises(Rejected):
                json_object(text)
        self.assertEqual(json_object('```json\n{"a":1}\n```'), {"a": 1})


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT / "experiments/frozen_inference/skills.json").read_text())

    def test_unseen_parameters_execute_not_lookup(self):
        library = SkillLibrary(self.payload)
        text = "A crate contains 11 packets of 13 items each. How many items are there?"
        result = Pipeline(skills=library).predict(text)
        self.assertEqual(result["prediction"], "143")
        self.assertEqual(result["route"], "microprogram")
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(library.fingerprint, SkillLibrary(copy.deepcopy(self.payload)).fingerprint)

    def test_domain_and_extra_facts_rejected(self):
        library = SkillLibrary(self.payload)
        self.assertIsNone(library.solve("A crate contains -3 packets of 7 items each. How many items are there?"))
        self.assertIsNone(library.solve("A crate contains 3 packets of 7 items each. One packet is missing. How many items are there?"))

    def test_bad_checks_code_and_test_origin_rejected(self):
        changes = [{"formula": "__import__('os')"}, {"formula": "a+b"}, {"bounds": {}}, {"positive": []}]
        for change in changes:
            payload = copy.deepcopy(self.payload)
            payload["skills"][0].update(change)
            with self.subTest(change=change), self.assertRaises(Rejected):
                SkillLibrary(payload)
        self.payload["origin"] = "test"
        with self.assertRaises(Rejected):
            SkillLibrary(self.payload)

    def test_no_regex_injection_and_ambiguous_hits_fail_closed(self):
        item = copy.deepcopy(self.payload["skills"][0])
        item["name"] = "alternative"
        item["formula"] = "a*b+(a-3)*(a-12)"
        self.payload["skills"].append(item)  # fits two tests, differs on new input
        library = SkillLibrary(self.payload)
        self.assertIsNone(library.solve("A crate contains 11 packets of 13 items each. How many items are there?"))
        # Placeholder parser permits only its literal DSL, never arbitrary regex.
        self.payload["skills"][0]["template"] = "{a:.*}"
        with self.assertRaises(Rejected):
            SkillLibrary(self.payload)


class PolicyTests(unittest.TestCase):
    def pairs(self, beneficial=True, n=40):
        base, new = [], []
        for i in range(n):
            common = {"case_id": f"synthetic/task-{i // 25}/{i % 25}", "index": i % 25,
                      "split": "calibration", "question_sha256": digest_text(str(i)),
                      "feature_bucket": features("a question")}
            base.append({**common, "correct": not beneficial, "elapsed_seconds": 0.2})
            new.append({**common, "correct": beneficial, "elapsed_seconds": 0.4, "compile_attempted": True})
        return base, new

    def test_beneficial_and_harmful_policies(self):
        for beneficial in [True, False]:
            policy = RepairPolicy(fit_policy(*self.pairs(beneficial)))
            self.assertEqual(policy.allows("a question"), beneficial)
            self.assertFalse(policy.allows("x" * 5000))

    def test_sparse_or_expensive_policy_preserves_direct(self):
        self.assertFalse(RepairPolicy(fit_policy(*self.pairs(n=2))).allows("q"))
        self.assertFalse(RepairPolicy(fit_policy(*self.pairs(), penalty_per_second=10)).allows("q"))

    def test_test_rows_duplicates_or_mismatches_rejected(self):
        base, new = self.pairs()
        new[0]["split"] = "test"
        with self.assertRaises(Rejected):
            fit_policy(base, new)
        base, new = self.pairs()
        new.append(new[0])
        with self.assertRaises(Rejected):
            fit_policy(base, new)
        base, new = self.pairs()
        new[0]["question_sha256"] = "0" * 64
        with self.assertRaises(Rejected):
            fit_policy(base, new)

    def test_fast_path_wins_do_not_calibrate_compilation(self):
        base, new = self.pairs()
        for row in new:
            row["compile_attempted"] = False
        with self.assertRaises(Rejected):
            fit_policy(base, new)

    def test_policy_can_skip_compiler(self):
        policy = RepairPolicy(fit_policy(*self.pairs(False)))
        client = FakeClient(["direct"])
        result = Pipeline(client, Config(enable_compile=True), policy=policy).predict("a question")
        self.assertEqual([c["action"] for c in result["calls"]], ["direct"])


class ClientAndCliTests(unittest.TestCase):
    def test_client_records_one_user_and_nested_usage_without_key(self):
        response = {"choices": [{"message": {"content": "A", "reasoning_content": "hidden"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 3}}}
        sent = []
        def fake_open(req, timeout):
            sent.append(json.loads(req.data))
            return io.BytesIO(json.dumps(response).encode())
        with patch("frozen_inference.backend.request.urlopen", fake_open):
            record = ChatClient("http://localhost:8889/v1", "test", "secret-key").complete("question", max_tokens=8, seed=1)
        self.assertEqual(sent[0]["messages"], [{"role": "user", "content": "question"}])
        self.assertNotIn("secret-key", json.dumps(record))
        self.assertEqual(record["message"]["reasoning_content"], "hidden")
        self.assertEqual(merge_usage([record, record])["completion_tokens_details"]["reasoning_tokens"], 6)

    def test_nonretryable_http_error_and_missing_usage(self):
        exc = HTTPError("http://localhost/v1", 400, "bad", {}, None)
        with patch("frozen_inference.backend.request.urlopen", side_effect=exc) as mock:
            with self.assertRaises(BackendError) as cm:
                ChatClient("http://localhost/v1", "test", retries=2).complete("q", max_tokens=8, seed=1)
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(len(cm.exception.record["attempts"]), 1)
        for url in ["file:///etc/passwd", "http://user:secret@localhost/v1", "https://host/v1?key=secret"]:
            with self.assertRaises(ValueError):
                ChatClient(url, "test")

    def test_retry_preserves_attempts_and_unknown_consumption(self):
        response = {"choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1}}
        with patch("frozen_inference.backend.request.urlopen", side_effect=[TimeoutError(), io.BytesIO(json.dumps(response).encode())]), \
                patch("frozen_inference.backend.time.sleep"):
            record = ChatClient("http://localhost/v1", "test", retries=1).complete("q", max_tokens=8, seed=1)
        self.assertEqual(len(record["attempts"]), 2)
        self.assertFalse(record["usage_complete"])

    def test_split_filters_and_duplicate_protection(self):
        rows = [{"benchmark": "b", "task": "t", "index": i, "input": "q"} for i in [0, 24, 25, 49, 50]]
        self.assertEqual([r["index"] for r in select_examples(rows, "calibration")], [0, 24])
        self.assertEqual([r["index"] for r in select_examples(rows, "validation")], [25, 49])
        self.assertEqual([r["index"] for r in select_examples(rows, "test")], [50])
        with self.assertRaises(Rejected):
            select_examples(rows + [rows[0]], "external")

    def test_offline_cli_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stdout", new_callable=io.StringIO):
            output = Path(tmp) / "run"
            argv = ["run", "--examples", str(ROOT / "experiments/frozen_inference/smoke.jsonl"),
                    "--skills", str(ROOT / "experiments/frozen_inference/skills.json"), "--offline", "--unscored",
                    "--study-id", "synthetic-smoke", "--split", "external", "--output-dir", str(output)]
            self.assertEqual(main(argv), 0)
            predictions = read_jsonl(output / "predictions.jsonl")
            self.assertEqual(len(predictions), 4)
            self.assertEqual(sum(r["error"] is not None for r in predictions), 1)
            self.assertEqual({r["prediction"] for r in predictions}, {"True", "42", "143", ""})
            self.assertTrue((output / "run_config.sha256").is_file())
            with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                main(argv)

    def test_test_split_requires_explicit_permission(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                main(["run", "--examples", str(ROOT / "experiments/frozen_inference/smoke.jsonl"),
                      "--offline", "--unscored", "--study-id", "new-study", "--split", "test",
                      "--output-dir", str(Path(tmp) / "run")])
            self.assertFalse((Path(tmp) / "run").exists())

    def test_missing_usage_not_reported_as_free_tokens(self):
        row = Pipeline(FakeClient([{"content": "A", "attempts": [{}]}])).predict("q")
        row.update(correct=False, benchmark="b", task="t")
        report = summary([row])
        self.assertIsNone(report["mean_completion_tokens"])
        self.assertFalse(report["usage_complete"])


if __name__ == "__main__":
    unittest.main()
