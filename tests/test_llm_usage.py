import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.api.llm_usage import calculate_llm_usage, summarize_llm_usage


class LLMUsageTests(unittest.TestCase):
    def test_uses_provider_tokens_and_known_model_pricing(self) -> None:
        response = AIMessage(
            content="Answer",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 200,
                "total_tokens": 1_200,
                "input_token_details": {"cache_read": 100},
            },
        )

        usage = calculate_llm_usage(
            [HumanMessage(content="Question")],
            response,
            "openai",
            "gpt-6-astra",
        )

        self.assertEqual(usage.token_count_source, "provider")
        self.assertEqual(usage.cached_input_tokens, 100)
        self.assertEqual(usage.total_tokens, 1_200)
        self.assertEqual(usage.pricing_source, "built_in")
        self.assertAlmostEqual(usage.estimated_cost_usd, 0.0191)

    def test_raw_openai_prompt_tokens_already_include_cached_tokens(self) -> None:
        response = AIMessage(
            content="Answer",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 200,
                    "total_tokens": 1_200,
                    "prompt_tokens_details": {"cached_tokens": 100},
                }
            },
        )

        usage = calculate_llm_usage(
            [HumanMessage(content="Question")],
            response,
            "openai",
            "gpt-6-astra",
        )

        self.assertEqual(usage.input_tokens, 1_000)
        self.assertEqual(usage.total_tokens, 1_200)
        self.assertAlmostEqual(usage.estimated_cost_usd, 0.0191)

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_tokens_but_no_cost_when_pricing_is_unknown(self) -> None:
        response = AIMessage(content="Four words in this answer")

        usage = calculate_llm_usage(
            [HumanMessage(content="A short question")],
            response,
            "huggingface",
            "google/gemma-2-2b-it",
        )

        self.assertEqual(usage.token_count_source, "estimated")
        self.assertGreater(usage.total_tokens, 0)
        self.assertIsNone(usage.estimated_cost_usd)
        self.assertEqual(usage.pricing_source, "unavailable")

    @patch.dict(
        os.environ,
        {
            "HF_INPUT_COST_PER_MILLION_USD": "0.10",
            "HF_OUTPUT_COST_PER_MILLION_USD": "0.20",
        },
        clear=True,
    )
    def test_supports_environment_pricing_for_provider_routed_models(self) -> None:
        response = AIMessage(
            content="Answer",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )

        usage = calculate_llm_usage(
            [HumanMessage(content="Question")],
            response,
            "huggingface",
            "google/gemma-2-2b-it",
        )

        self.assertEqual(usage.pricing_source, "environment")
        self.assertAlmostEqual(usage.estimated_cost_usd, 0.00002)

    def test_aggregate_exposes_unpriced_calls(self) -> None:
        priced = calculate_llm_usage(
            [HumanMessage(content="Question")],
            AIMessage(
                content="Answer",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            ),
            "grok",
            "grok-4.7",
        )
        unpriced = calculate_llm_usage(
            [HumanMessage(content="Question")],
            AIMessage(content="Answer"),
            "huggingface",
            "unknown-model",
        )

        summary = summarize_llm_usage([priced, unpriced])

        self.assertEqual(summary.llm_calls, 2)
        self.assertEqual(summary.priced_calls, 1)
        self.assertEqual(summary.unpriced_calls, 1)
        self.assertEqual(summary.estimated_cost_usd, priced.estimated_cost_usd)


if __name__ == "__main__":
    unittest.main()
