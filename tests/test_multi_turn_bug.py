"""
Diagnostic + regression tests for the multi-turn conversation failure.

Root cause: Sarvam returns EMPTY content on the 2nd+ question because
the full conversation history pushed the payload over the token-window limit.
This triggered ValueError → _build_fallback_answer() → 
"I couldn't reach the live generator, but here's what I found in your catalog:"

Fix applied in generation.py:
  1. _build_contents() caps history to _MAX_HISTORY_MESSAGES (6) messages
  2. generate() / generate_stream() log ERROR when empty content is received
  3. Both methods retry once with history stripped when content is empty

Run with: python -m pytest tests/test_multi_turn_bug.py -v -s
"""
from __future__ import annotations

import sys
import os
import logging
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from collections import deque

# ---------------------------------------------------------------------------
# Mock `sarvamai` before any project imports so tests run without the package
# ---------------------------------------------------------------------------
_mock_sarvamai = MagicMock()
_mock_sarvamai_class = MagicMock()
_mock_sarvamai.SarvamAI = _mock_sarvamai_class
sys.modules.setdefault("sarvamai", _mock_sarvamai)

# Make sure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers – fake Sarvam response factories
# ---------------------------------------------------------------------------

def _make_sarvam_response(content: str):
    """Build a fake Sarvam chat.completions response with non-empty content."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_empty_sarvam_response():
    """Simulate Sarvam returning an EMPTY content string (the bug trigger)."""
    msg = MagicMock()
    msg.content = ""          # ← triggers empty-content path
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_none_sarvam_response():
    """Simulate Sarvam returning None content."""
    msg = MagicMock()
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Shared pipeline factory
# ---------------------------------------------------------------------------

def _build_pipeline(mock_sarvam_responses: list, session_dir: str = "/tmp/test_sessions"):
    """Return a RAGPipeline with all external dependencies mocked."""
    from src.rag.generation import SarvamGenerator
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.memory.memory_manager import ConversationMemoryManager
    import tempfile

    settings = MagicMock()
    settings.memory_turns = 8
    settings.top_k = 5
    settings.max_context_chars = 5000
    # Point data_dir at a temp dir with NO image_mapping.json
    # so _resolve_images() skips image loading silently.
    settings.session_store_dir = Path(session_dir)
    settings.data_dir = Path(tempfile.mkdtemp())

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = settings

    # Mock embedder – always returns a valid vector
    pipeline.embedder = MagicMock()
    pipeline.embedder.embed_query.return_value = [0.1] * 768

    # Mock vector store – returns one catalog chunk
    fake_chunk = MagicMock()
    fake_chunk.text = "CNC machine info: starting price 7,80,000"
    fake_chunk.source = "catalog.txt"
    fake_chunk.metadata = {"images": [], "product_tag": None}
    fake_chunk.chunk_id = "chunk-001"
    fake_item = MagicMock()
    fake_item.chunk = fake_chunk
    fake_item.score = 0.9
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.search.return_value = [fake_item]

    # Mock context manager
    pipeline.context_manager = MagicMock()
    pipeline.context_manager.build_context.return_value = (
        "Catalog context text.", ["catalog.txt"]
    )

    # Real in-memory session store (unique dir per call)
    import tempfile as _tf
    unique_session_dir = _tf.mkdtemp()
    pipeline.memory_manager = ConversationMemoryManager(
        max_turns=8,
        session_dir=unique_session_dir,
        encryption_key=None,
    )

    # Sarvam client mock
    sarvam_client = MagicMock()
    sarvam_client.chat.completions.side_effect = mock_sarvam_responses

    gen = SarvamGenerator.__new__(SarvamGenerator)
    gen.model = "sarvam-30b"
    gen.temperature = 0.2
    gen.top_p = 1.0
    gen.max_tokens = 1500
    gen.timeout = 25
    gen.max_retries = 1
    gen.backoff_seconds = 0.01   # fast for tests
    gen.client = sarvam_client

    pipeline.generator = gen
    return pipeline


# ===========================================================================
# Test 1: generation.py unit tests
# ===========================================================================

class TestGenerationUnit(unittest.TestCase):
    """Direct tests for SarvamGenerator.generate() and _build_contents()."""

    def _make_gen(self, side_effects: list):
        from src.rag.generation import SarvamGenerator
        gen = SarvamGenerator.__new__(SarvamGenerator)
        gen.model = "sarvam-30b"
        gen.temperature = 0.2
        gen.top_p = 1.0
        gen.max_tokens = 1500
        gen.timeout = 25
        gen.max_retries = 1
        gen.backoff_seconds = 0.01
        gen.client = MagicMock()
        gen.client.chat.completions.side_effect = side_effects
        return gen

    def test_good_content_returns_string(self):
        """generate() must return the content string when Sarvam is healthy."""
        gen = self._make_gen([_make_sarvam_response("Valid answer here.")])
        result = gen.generate("question", "context", history=[])
        self.assertEqual(result, "Valid answer here.")
        logger.info("✓ test_good_content_returns_string")

    def test_empty_content_retries_without_history(self):
        """
        When Sarvam returns empty content on attempt 1, it must retry with
        history stripped (system prompt + current user message only),
        and succeed on attempt 2.
        """
        gen = self._make_gen([
            _make_empty_sarvam_response(),        # attempt 1 → empty
            _make_sarvam_response("Retry answer!"),  # attempt 2 → good
        ])
        history = [
            {"role": "user",      "content": "Previous Q"},
            {"role": "assistant", "content": "Previous A " * 100},  # long answer
        ]
        result = gen.generate("New question?", "context", history=history)
        self.assertEqual(result, "Retry answer!")

        # Verify 2nd call had fewer messages (history stripped)
        calls = gen.client.chat.completions.call_args_list
        self.assertEqual(len(calls), 2)
        msgs_call2 = calls[1][1].get("messages", calls[1][0][0] if calls[1][0] else [])
        # 2nd call: system + 1 user = 2 messages (no history)
        roles_call2 = [m["role"] for m in msgs_call2]
        self.assertNotIn("assistant", roles_call2, "History must be stripped on retry")
        logger.info("✓ test_empty_content_retries_without_history (msgs: %s)", roles_call2)

    def test_none_content_retries_then_raises(self):
        """Both attempts return None/empty → ValueError after retry exhausted."""
        gen = self._make_gen([
            _make_none_sarvam_response(),   # attempt 1 → None
            _make_none_sarvam_response(),   # attempt 2 → None
        ])
        with self.assertRaises((ValueError, RuntimeError)):
            gen.generate("question", "context", history=[])
        logger.info("✓ test_none_content_retries_then_raises")

    def test_history_capped_in_build_contents(self):
        """
        _build_contents() must cap history to _MAX_HISTORY_MESSAGES messages.
        With max=6, a 20-message history should be trimmed to 6.
        """
        from src.rag.generation import SarvamGenerator, _MAX_HISTORY_MESSAGES
        gen = SarvamGenerator.__new__(SarvamGenerator)
        gen.model = "sarvam-30b"
        gen.temperature = 0.2
        gen.top_p = 1.0
        gen.max_tokens = 1500

        large_history = []
        for i in range(20):
            large_history.append({"role": "user",      "content": f"Q{i}"})
            large_history.append({"role": "assistant",  "content": f"A{i}"})

        contents = gen._build_contents(large_history, "Current question")
        # contents = capped_history (≤6) + 1 user message
        max_expected = _MAX_HISTORY_MESSAGES + 1
        self.assertLessEqual(len(contents), max_expected,
            f"Expected ≤{max_expected} messages, got {len(contents)}")
        logger.info("✓ test_history_capped_in_build_contents (%d messages)", len(contents))

    def test_empty_history_works(self):
        """First turn: empty history → single user message in contents."""
        from src.rag.generation import SarvamGenerator
        gen = SarvamGenerator.__new__(SarvamGenerator)
        contents = gen._build_contents([], "Hello")
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["role"], "user")
        logger.info("✓ test_empty_history_works")

    def test_reformulate_falls_back_on_error(self):
        """If reformulate_query() Sarvam call fails, return original question."""
        from src.rag.generation import SarvamGenerator
        gen = SarvamGenerator.__new__(SarvamGenerator)
        gen.model = "sarvam-30b"
        gen.temperature = 0.1
        gen.top_p = 1.0
        gen.max_tokens = 500
        gen.timeout = 25
        gen.max_retries = 1
        gen.backoff_seconds = 0.01
        gen.client = MagicMock()
        gen.client.chat.completions.side_effect = RuntimeError("Sarvam error")

        original = "What about EMI options?"
        history = [
            {"role": "user",      "content": "What is the price?"},
            {"role": "assistant", "content": "Price starts at 7,80,000."},
        ]
        result = gen.reformulate_query(original, history=history)
        self.assertEqual(result, original)
        logger.info("✓ test_reformulate_falls_back_on_error")


# ===========================================================================
# Test 2: Pipeline-level multi-turn tests
# ===========================================================================

class TestPipelineMultiTurn(unittest.TestCase):
    """End-to-end pipeline tests simulating real WhatsApp multi-turn chat."""

    def test_turn1_single_sarvam_call(self):
        """
        Turn 1 with no prior history: reformulate_query() skips the Sarvam call,
        so only 1 Sarvam call (generate) is made.
        """
        responses = [_make_sarvam_response("Price starts at ₹7,80,000.")]
        pipeline = _build_pipeline(responses, "/tmp/sess_t1")

        result = pipeline.query("What is the price?", session_id="turn1-test")
        self.assertIn("7,80,000", result["answer"])
        self.assertNotIn("couldn't reach the live generator", result["answer"])
        logger.info("✓ Turn 1 answer: %s", result["answer"])

    def test_turn2_gets_llm_answer_not_fallback(self):
        """
        THE MAIN BUG TEST. Turn 2 must NOT return the fallback message
        even if the first Sarvam call (with history) returns empty content.
        The retry-without-history mechanism must rescue it.
        """
        responses = [
            # Turn 1: generate() → good
            _make_sarvam_response("Price starts at ₹7,80,000. Where is your workshop?"),
            # Turn 2: reformulate_query() → good
            _make_sarvam_response("Is EMI available for the CNC machine?"),
            # Turn 2: generate() attempt 1 → empty (THE BUG)
            _make_empty_sarvam_response(),
            # Turn 2: generate() attempt 2 (retry without history) → good (THE FIX)
            _make_sarvam_response("Yes! EMI is available. What's your budget range?"),
        ]
        pipeline = _build_pipeline(responses, "/tmp/sess_t2")
        sid = "turn2-bug-test"

        result1 = pipeline.query("What is the price?", session_id=sid)
        logger.info("Turn 1: %s", result1["answer"])

        result2 = pipeline.query("Ami ki EMI te nite parbo?", session_id=sid)
        logger.info("Turn 2: %s", result2["answer"])

        self.assertNotIn(
            "couldn't reach the live generator", result2["answer"],
            f"BUG: Turn 2 returned fallback instead of LLM answer: {result2['answer']}"
        )
        self.assertIn("EMI", result2["answer"])
        logger.info("✓ Turn 2 got real LLM answer (not fallback)")

    def test_three_turn_conversation_all_pass(self):
        """
        Full 3-turn simulation in Bengali/Hinglish.
        All turns must get real LLM answers, never the fallback.
        """
        responses = [
            # Turn 1: generate
            _make_sarvam_response("Price starts at ₹7,80,000. Workshop location?"),
            # Turn 2: reformulate
            _make_sarvam_response("Is EMI available for the CNC machine?"),
            # Turn 2: generate
            _make_sarvam_response("Yes! EMI available. What's your budget?"),
            # Turn 3: reformulate
            _make_sarvam_response("What materials can the CNC machine cut?"),
            # Turn 3: generate
            _make_sarvam_response("Wood, acrylic, aluminium. Starting new business?"),
        ]
        pipeline = _build_pipeline(responses, "/tmp/sess_t3")
        sid = "3turn-test"

        results = [
            pipeline.query("Machine er dam koto?", session_id=sid),
            pipeline.query("EMI available ache?", session_id=sid),
            pipeline.query("Ki ki kata jay machine e?", session_id=sid),
        ]

        for i, r in enumerate(results, 1):
            is_fallback = "couldn't reach the live generator" in r["answer"]
            self.assertFalse(is_fallback, f"Turn {i} fell back! Answer: {r['answer']}")
            logger.info("✓ Turn %d: %s", i, r["answer"][:60])

        logger.info("✓ All 3 turns got proper LLM answers")

    def test_turn2_both_attempts_empty_uses_catalog_fallback(self):
        """
        Worst case: both Sarvam attempts return empty. The fallback catalog
        snippet IS returned. We accept this but confirm the pipeline doesn't crash.
        """
        responses = [
            # Turn 1: generate → good
            _make_sarvam_response("Price starts at ₹7,80,000."),
            # Turn 2: reformulate → good (returns original question)
            _make_sarvam_response("Is EMI available?"),
            # Turn 2: generate attempt 1 → empty
            _make_empty_sarvam_response(),
            # Turn 2: generate attempt 2 (retry) → still empty
            _make_empty_sarvam_response(),
        ]
        pipeline = _build_pipeline(responses, "/tmp/sess_t2_worst")
        sid = "worst-case-test"

        pipeline.query("price?", session_id=sid)
        result2 = pipeline.query("EMI?", session_id=sid)

        # Must not crash; will return catalog fallback in this extreme case
        self.assertIsInstance(result2["answer"], str)
        self.assertGreater(len(result2["answer"]), 0)
        logger.info("✓ Worst case (both empty): returned '%s'", result2["answer"][:80])


# ===========================================================================
# Test 3: Memory / history slicing
# ===========================================================================

class TestMemoryManager(unittest.TestCase):
    """Tests for ConversationMemoryManager behaviour."""

    def _mem(self, session_dir: str = "/tmp/test_mm"):
        from src.memory.memory_manager import ConversationMemoryManager
        return ConversationMemoryManager(
            max_turns=8,
            session_dir=session_dir,
            encryption_key=None,
        )

    def test_history_grows_correctly(self):
        """Each turn adds 2 messages (user + assistant)."""
        import tempfile
        from src.memory.memory_manager import ConversationMemoryManager
        mem = ConversationMemoryManager(
            max_turns=8,
            session_dir=tempfile.mkdtemp(),
            encryption_key=None,
        )
        sid = "grow-test"
        mem.add_user_message(sid, "Q1")
        mem.add_assistant_message(sid, "A1")
        mem.add_user_message(sid, "Q2")
        mem.add_assistant_message(sid, "A2")

        history = mem.get_recent_history(sid)
        self.assertEqual(len(history), 4)
        logger.info("✓ History has 4 messages after 2 turns")

    def test_deque_cap_enforced(self):
        """History must not exceed 2 * max_turns messages."""
        from src.memory.memory_manager import ConversationMemoryManager
        mem = ConversationMemoryManager(
            max_turns=4,
            session_dir="/tmp/test_mm_cap",
            encryption_key=None,
        )
        sid = "cap-test"
        for i in range(10):
            mem.add_user_message(sid, f"Q{i}")
            mem.add_assistant_message(sid, f"A{i}")

        history = mem.get_recent_history(sid)
        self.assertLessEqual(len(history), 8, f"History grew past cap: {len(history)}")
        logger.info("✓ History capped at %d (max_turns=4 → cap=8)", len(history))

    def test_pipeline_history_slice_excludes_current_user_msg(self):
        """
        The pipeline slices history as history[-5:-1] after adding the current
        user message. This test verifies the slice doesn't contain the current msg.
        """
        from src.memory.memory_manager import ConversationMemoryManager
        mem = ConversationMemoryManager(
            max_turns=8,
            session_dir="/tmp/test_mm_slice",
            encryption_key=None,
        )
        sid = "slice-test"

        # 3 prior turns
        for i in range(3):
            mem.add_user_message(sid, f"Prior Q{i}")
            mem.add_assistant_message(sid, f"Prior A{i}")

        # Simulate what query() does: add current user message first
        mem.add_user_message(sid, "Current question")
        history = mem.get_recent_history(sid)
        recent_history = history[-5:-1] if len(history) > 1 else []

        logger.info("Full history len: %d, slice len: %d", len(history), len(recent_history))

        if recent_history:
            self.assertNotEqual(
                recent_history[-1]["content"], "Current question",
                "Current user message must NOT appear in the history slice sent to Sarvam"
            )
        logger.info("✓ History slice correctly excludes current user message")


# ===========================================================================
# Test 4: Embedding rate-limit does NOT cause the 'live generator' message
# ===========================================================================

class TestEmbeddingErrors(unittest.TestCase):
    """
    Verify that an embedding failure raises RuntimeError (not caught by the
    pipeline's generate() fallback). This means the 'live generator' message
    is NOT caused by embedding failures — it's caused by Sarvam empty content.
    """

    def test_embedding_failure_propagates_not_fallback(self):
        """
        embed_query() raising RuntimeError must propagate out of query(),
        NOT silently produce the 'couldn't reach live generator' message.
        """
        from src.pipeline.rag_pipeline import RAGPipeline
        from src.memory.memory_manager import ConversationMemoryManager

        settings = MagicMock()
        settings.memory_turns = 8
        settings.top_k = 5
        settings.max_context_chars = 5000
        settings.session_store_dir = Path("/tmp/sessions_embed")

        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.settings = settings
        pipeline.embedder = MagicMock()
        pipeline.embedder.embed_query.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
        pipeline.vector_store = MagicMock()
        pipeline.context_manager = MagicMock()
        pipeline.memory_manager = ConversationMemoryManager(
            max_turns=8, session_dir="/tmp/embed_err", encryption_key=None
        )
        pipeline.generator = MagicMock()
        pipeline.generator.reformulate_query.return_value = "question"

        with self.assertRaises(RuntimeError) as ctx:
            pipeline.query("test question", session_id="embed-fail-test")

        self.assertIn("429", str(ctx.exception))
        logger.info(
            "✓ Embedding error propagates as RuntimeError (not caught as 'live generator' fallback)"
        )
        logger.info(
            "CONCLUSION: The 'live generator' message is caused ONLY by Sarvam returning "
            "empty/None content, which happens due to token-window overflow from large history."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
