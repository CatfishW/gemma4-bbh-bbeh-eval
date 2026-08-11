from pathlib import Path
import tempfile
import unittest

from eval_benchmarks import Example
from rl.eval_official_thinking import (
    EvalRequest,
    acquire_output_lock,
    plan_batches,
    select_examples,
)
from rl.official_thinking import (
    BBEH_EVALUATION_SUFFIX,
    detect_huggingface_revision,
    native_thinking_prompt_token_ids,
    official_bbeh_prompt,
    parse_native_thinking_response,
    stable_batch_seed,
    trim_generated_token_ids,
    validate_native_thinking_prompt,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_call = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = (messages, kwargs)
        return [2, 4, 98, 5]

    def decode(self, token_ids, skip_special_tokens=False):
        if token_ids == [2, 4, 98, 5]:
            return (
                "<bos><|turn>system\n<|think|>\n<turn|>\n"
                "<|turn>user\nQuestion<turn|>\n<|turn>model\n"
            )
        return "<|channel>thought\nwork<channel|>answer<turn|>"

    def convert_tokens_to_ids(self, token):
        return 98 if token == "<|think|>" else -1

    def parse_response(self, response):
        return {"role": "assistant", "thinking": "work", "content": "answer"}


class PromptAndParsingTests(unittest.TestCase):
    def test_official_bbeh_prompt_appends_published_answer_contract(self):
        prompt = official_bbeh_prompt("Question\n")
        self.assertEqual(prompt, "Question\n\n" + BBEH_EVALUATION_SUFFIX)
        self.assertIn('prefix "The answer is:"', prompt)
        self.assertIn("Think step by step", prompt)

    def test_native_prompt_uses_one_user_message_and_template_thinking(self):
        tokenizer = FakeTokenizer()
        token_ids = native_thinking_prompt_token_ids(tokenizer, "Question")

        self.assertEqual(token_ids, [2, 4, 98, 5])
        messages, kwargs = tokenizer.template_call
        self.assertEqual(messages, [{"role": "user", "content": "Question"}])
        self.assertTrue(kwargs["enable_thinking"])
        self.assertTrue(kwargs["add_generation_prompt"])
        validate_native_thinking_prompt(tokenizer, token_ids)

    def test_parser_scores_only_final_content(self):
        parsed = parse_native_thinking_response(FakeTokenizer(), [7, 8, 9])
        self.assertEqual(parsed.thinking, "work")
        self.assertEqual(parsed.prediction, "answer")
        self.assertNotIn("work", parsed.prediction)
        self.assertIsNone(parsed.parse_error)

    def test_parser_fails_closed(self):
        tokenizer = FakeTokenizer()
        tokenizer.parse_response = lambda response: (_ for _ in ()).throw(ValueError("bad"))
        parsed = parse_native_thinking_response(tokenizer, [7])
        self.assertEqual(parsed.prediction, "")
        self.assertIn("ValueError", parsed.parse_error)

    def test_missing_final_content_is_counted_as_parse_failure(self):
        tokenizer = FakeTokenizer()
        tokenizer.parse_response = lambda response: {
            "role": "assistant",
            "thinking": "unfinished",
        }
        parsed = parse_native_thinking_response(tokenizer, [7])
        self.assertEqual(parsed.prediction, "")
        self.assertEqual(parsed.parse_error, "missing final content")


class GenerationAccountingTests(unittest.TestCase):
    def test_trim_at_first_stop_before_batch_padding(self):
        result = trim_generated_token_ids(
            [10, 11, 106, 0, 0],
            stop_token_ids=[1, 50, 106],
            pad_token_id=0,
            max_new_tokens=8,
        )
        self.assertEqual(result.token_ids, [10, 11, 106])
        self.assertTrue(result.stopped)
        self.assertEqual(result.stop_token_id, 106)
        self.assertFalse(result.truncated)

    def test_max_length_without_stop_is_truncated(self):
        result = trim_generated_token_ids(
            [10, 11, 12],
            stop_token_ids=[1, 106],
            pad_token_id=0,
            max_new_tokens=3,
        )
        self.assertFalse(result.stopped)
        self.assertTrue(result.truncated)

    def test_batch_seed_is_stable_and_batch_specific(self):
        seed = stable_batch_seed(7, ["bbeh/a/0", "bbeh/a/1"])
        self.assertEqual(seed, stable_batch_seed(7, ["bbeh/a/0", "bbeh/a/1"]))
        self.assertNotEqual(seed, stable_batch_seed(7, ["bbeh/a/0"]))


class ScopeAndBatchTests(unittest.TestCase):
    def test_frozen_scope_starts_at_index_50(self):
        examples = [Example("bbeh", "task", index, "q", "a") for index in range(55)]
        self.assertEqual(len(select_examples(examples, "all")), 55)
        frozen = select_examples(examples, "frozen_test")
        self.assertEqual([example.index for example in frozen], [50, 51, 52, 53, 54])

    def test_batch_planner_respects_count_and_token_budget(self):
        example = Example("bbeh", "task", 0, "q", "a")
        requests = [
            EvalRequest(example, f"bbeh/task/{index}", [1] * length)
            for index, length in enumerate((10, 11, 12, 30))
        ]
        batches = plan_batches(
            requests,
            batch_size=3,
            max_batch_tokens=75,
            max_new_tokens=10,
        )
        self.assertEqual([len(batch) for batch in batches], [3, 1])
        for batch in batches:
            longest = max(len(request.prompt_token_ids) for request in batch)
            self.assertLessEqual((longest + 10) * len(batch), 75)


class RevisionTests(unittest.TestCase):
    def test_detects_download_metadata_revision(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            metadata = model / ".cache" / "huggingface" / "download" / "config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(f"{revision}\nblob\ntimestamp\n")
            self.assertEqual(detect_huggingface_revision(model), revision)


class OutputLockTests(unittest.TestCase):
    def test_rejects_a_second_writer_and_releases_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = acquire_output_lock(output)
            try:
                with self.assertRaisesRegex(RuntimeError, "another evaluator"):
                    acquire_output_lock(output)
            finally:
                first.close()
            second = acquire_output_lock(output)
            second.close()


if __name__ == "__main__":
    unittest.main()
