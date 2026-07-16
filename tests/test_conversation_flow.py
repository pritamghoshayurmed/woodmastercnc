import json
import os
import unittest

# The application supports a local development flow when persistence is disabled.
os.environ["DB_ENABLED"] = "false"

from src.messenger.conversation_flow import ConversationFlowManager


class ConversationFlowTests(unittest.TestCase):
    def setUp(self):
        self.extractions = {
            "People call me Mirana Mir.": "Mirana Mir",
            "Actually, I go by Amar — nice to meet you.": "Amar",
            "मुझे लोग अमर कहते हैं।": "अमर",
        }
        self.flow = ConversationFlowManager(
            name_extractor=lambda message, _language: self.extractions.get(message)
        )
        self.session = "whatsapp:919999999999"

    def _select_english(self, session=None):
        response = self.flow.handle_message(session or self.session, "English")
        self.assertEqual(response.reply, "Please tell me your name.")

    def test_llm_extracted_names_advance_for_natural_phrasings(self):
        self._select_english()
        for index, (message, expected_name) in enumerate(self.extractions.items()):
            with self.subTest(message=message):
                session = f"whatsapp:919999999{index}"
                self._select_english(session)
                response = self.flow.handle_message(session, message)
                self.assertEqual(response.user_name, expected_name)
                self.assertIn(expected_name.split()[0], response.reply)
                self.assertIn("city or address", response.reply)

    def test_no_name_does_not_advance_or_corrupt_state(self):
        self._select_english()
        response = self.flow.handle_message(self.session, "Can you tell me the price of a CNC machine?")
        self.assertEqual(response.reply, "Please tell me your name.")
        self.assertIsNone(response.user_name)
        self.assertEqual(self.flow._local[self.session]["stage"], "name")

    def test_name_address_then_chatting_flow(self):
        self._select_english()
        name = self.flow.handle_message(self.session, "People call me Mirana Mir.")
        self.assertEqual(name.user_name, "Mirana Mir")
        welcome = self.flow.handle_message(self.session, "Arambagh, Hooghly")
        self.assertIn("Mirana", welcome.reply)
        self.assertEqual(self.flow._local[self.session]["stage"], "chatting")

    def test_plain_name_advances_without_an_llm_extractor(self):
        self._select_english()
        response = self.flow.handle_message(self.session, "Pritam Ghosh")
        self.assertEqual(response.user_name, "Pritam Ghosh")
        self.assertEqual(response.stage, "address")
        self.assertIn("Pritam", response.reply)

    def test_strict_json_name_contract_accepts_unicode_and_rejects_invalid_values(self):
        self.assertEqual(
            ConversationFlowManager._parse_name_response(json.dumps({"name": "  Amar  Das "})),
            "Amar Das",
        )
        self.assertEqual(
            ConversationFlowManager._parse_name_response(json.dumps({"name": "মিরানা মীর"}, ensure_ascii=False)),
            "মিরানা মীর",
        )
        self.assertIsNone(ConversationFlowManager._parse_name_response('{"name": null}'))
        self.assertIsNone(ConversationFlowManager._parse_name_response("not json"))
        self.assertIsNone(ConversationFlowManager._parse_name_response('{"name": 12}'))


if __name__ == "__main__":
    unittest.main()
