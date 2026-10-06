import unittest

from logic import MODEL_VARIANTS, build_payload, response_parts, split_thinking


class SparkLogicTests(unittest.TestCase):
    def test_default_payload_is_direct_and_uses_selected_file_stem(self):
        payload = build_payload(
            variant="Spark 1.7B · fastest",
            preset="Balanced",
            thinking=False,
            prompt="Привет",
        )
        self.assertEqual(payload["model"], "Spark-X2.5-1.7B-q4mix")
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["messages"][-1]["content"], "Привет")

    def test_history_is_bounded_and_invalid_entries_are_ignored(self):
        history = [{"role": "user", "content": str(i)} for i in range(20)]
        history.append({"role": "system", "content": "must not pass"})
        payload = build_payload(
            variant="Spark 4B · quality",
            preset="Fast",
            thinking=True,
            prompt="go",
            history=history,
        )
        self.assertEqual(len(payload["messages"]), 14)
        self.assertEqual(payload["messages"][1]["content"], "8")
        self.assertEqual(payload["messages"][-2]["content"], "19")
        self.assertTrue(payload["enable_thinking"])

    def test_thinking_block_is_split_without_losing_answer(self):
        thought, answer = split_thinking("<think>check facts</think>Final answer")
        self.assertEqual(thought, "check facts")
        self.assertEqual(answer, "Final answer")

    def test_tool_only_response_is_preserved(self):
        content, calls, usage = response_parts(
            {
                "choices": [{"message": {"content": None, "tool_calls": [{"id": "call_1"}]}}],
                "usage": {"total_tokens": 12},
            }
        )
        self.assertEqual(content, "")
        self.assertEqual(calls, [{"id": "call_1"}])
        self.assertEqual(usage["total_tokens"], 12)

    def test_model_catalog_has_public_cmf_files(self):
        self.assertEqual(len(MODEL_VARIANTS), 4)
        self.assertTrue(all(item["file"].endswith(".cmf") for item in MODEL_VARIANTS.values()))


if __name__ == "__main__":
    unittest.main()
