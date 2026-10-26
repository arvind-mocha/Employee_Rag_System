import unittest

from langchain_core.messages import AIMessage, HumanMessage

from app.api.prompts import build_answer_messages, build_tool_selection_messages
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

    def test_places_history_before_the_current_grounded_question(self) -> None:
        chunk = RetrievedChunk(
            content="Part-time leave is prorated.",
            relevance_score=0.9,
            chunk_id="chunk-2",
            source_file="handbook.md",
            source_type="markdown",
            section="Vacation",
            chunk_index=2,
        )

        messages = build_answer_messages(
            "What about part-time employees?",
            [chunk],
            [("What is the vacation policy?", "Employees receive 15 days.")],
        )

        self.assertIsInstance(messages[1], HumanMessage)
        self.assertIsInstance(messages[2], AIMessage)
        self.assertIn("Part-time leave is prorated", messages[-1].content)
        self.assertIn("prior assistant answers", messages[0].content)

    def test_places_history_before_the_current_tool_decision(self) -> None:
        messages = build_tool_selection_messages(
            "Does that apply to contractors?",
            [("What is the leave policy?", "Employees receive paid leave.")],
        )

        self.assertEqual(messages[1].content, "What is the leave policy?")
        self.assertEqual(messages[2].content, "Employees receive paid leave.")
        self.assertEqual(messages[-1].content, "Does that apply to contractors?")


if __name__ == "__main__":
    unittest.main()
