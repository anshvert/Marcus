import json
import os
import unittest
from unittest.mock import patch

from marcus.agents.manager import OpenRouterBrain
from marcus.providers.base import chat_configuration


class Stream:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        event = {"choices": [{"delta": {"content": '{"reply":"Local reply","reflect":false}'}, "finish_reason": "stop"}]}
        return iter([("data: " + json.dumps(event) + "\n").encode(), b"data: [DONE]\n"])


class RecordingTransport:
    def open(self, request, **kwargs):
        self.request = request
        return Stream()


class ProviderTests(unittest.TestCase):
    @patch.dict(os.environ, {"MARCUS_CHAT_API_BASE": "http://127.0.0.1:1234/v1", "OPENROUTER_API_KEY": "must-not-leak"}, clear=True)
    def test_local_adapter_omits_cloud_credentials_and_provider_parameters(self):
        transport = RecordingTransport()
        brain = OpenRouterBrain(model="local-model", transport=transport)
        response = brain.respond("hi", history=[], memories=[])
        self.assertEqual(response.reply, "Local reply")
        self.assertIsNone(transport.request.get_header("Authorization"))
        self.assertNotIn("provider", json.loads(transport.request.data))

    @patch.dict(os.environ, {}, clear=True)
    def test_remote_plain_http_is_rejected(self):
        with self.assertRaises(ValueError):
            chat_configuration("key", "http://untrusted.example/v1/chat/completions")
