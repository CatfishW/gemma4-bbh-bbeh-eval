import unittest

from scripts.audit_output_formats import (
    exact_answer_schema,
    parse_raw_json,
    remove_single_code_fence,
)


class OutputFormatAuditTests(unittest.TestCase):
    def test_raw_schema_rejects_fences_and_extra_keys(self) -> None:
        self.assertTrue(exact_answer_schema(parse_raw_json('{"answer":"A"}')))
        self.assertFalse(exact_answer_schema(parse_raw_json('```json\n{"answer":"A"}\n```')))
        self.assertFalse(exact_answer_schema(parse_raw_json('{"answer":"A","why":"x"}')))

    def test_recovery_removes_one_code_fence(self) -> None:
        recovered = remove_single_code_fence('```json\n{"answer":"A"}\n```')
        self.assertTrue(exact_answer_schema(parse_raw_json(recovered)))


if __name__ == "__main__":
    unittest.main()
