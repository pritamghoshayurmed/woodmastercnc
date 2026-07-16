from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.memory.memory_manager import ConversationMemoryManager


class DatabaseMemoryTests(unittest.TestCase):
    def test_database_mode_reads_current_conversation_without_creating_a_file(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.memory.memory_manager.is_db_enabled", return_value=True
        ), patch(
            "src.memory.memory_manager.db_users.get_user_by_phone",
            return_value={"id": "user-1"},
        ), patch(
            "src.memory.memory_manager.db_conversations.get_active_conversation",
            return_value={"id": "conversation-1"},
        ), patch(
            "src.memory.memory_manager.db_messages.get_conversation_messages",
            return_value=[
                {"sender": "USER", "message": "Which model is suitable?"},
                {"sender": "BOT", "message": "How many chairs per day?"},
                {"sender": "USER", "message": "150 chairs."},
            ],
        ):
            memory = ConversationMemoryManager(session_dir=temp_dir)
            history = memory.get_context_window("whatsapp:919999999999", "150 chairs.")
            memory.add_assistant_message("whatsapp:919999999999", "Recommendation")

            self.assertEqual(
                history,
                [
                    {"role": "user", "content": "Which model is suitable?"},
                    {"role": "assistant", "content": "How many chairs per day?"},
                ],
            )
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_local_mode_remains_available_for_explicit_development_use(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.memory.memory_manager.is_db_enabled", return_value=False
        ):
            memory = ConversationMemoryManager(session_dir=temp_dir)
            memory.add_user_message("web:test", "Hello")
            self.assertEqual(memory.get_recent_history("web:test"), [{"role": "user", "content": "Hello"}])
            self.assertEqual(len(list(Path(temp_dir).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
