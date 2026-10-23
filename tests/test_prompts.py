import unittest

from app.api.prompts import build_answer_messages
from app.retrieval.models import RetrievedChunk


class AnswerPromptTests(unittest.TestCase):
    def test_numbers_sources_and_treats_context_as_untrusted(self) -> None:
        chunk = RetrievedChunk(
            content="Employees receive 15 vacation days.",
            relevance_score=0.9,
            chunk_id="chunk-1",
            source_file="handbook.md",
            source_type="markdown",
            section="Vacation",
            chunk_index=1,
        )

        messages = build_answer_messages("How many days?", [chunk])

        self.assertIn("untrusted", messages[0].content)
        self.assertIn("[1] Source", messages[1].content)
        self.assertIn("handbook.md", messages[1].content)
        self.assertIn("How many days?", messages[1].content)


if __name__ == "__main__":
    unittest.main()
