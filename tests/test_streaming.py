import json
import time
import unittest

from marcus.agents.manager import OpenRouterBrain, ReplyDeltaExtractor


class ReplyDeltaExtractorTests(unittest.TestCase):
    def test_emoji_surrogate_pair_is_not_emitted_as_invalid_unicode(self):
        extractor = ReplyDeltaExtractor()
        self.assertEqual(extractor.feed('{"reply":"Hi \\ud83d'), "Hi ")
        self.assertEqual(extractor.feed('\\ude01","reflect":false}'), "😁")

    def test_extracts_reply_across_fragments_and_decodes_escapes(self) -> None:
        extractor = ReplyDeltaExtractor()
        output = []

        output.append(extractor.feed('{"rep'))
        output.append(extractor.feed('ly":"Hello'))
        output.append(extractor.feed(' \\'))
        output.append(extractor.feed('"Marcus\\nnext","reflect":false}'))

        self.assertEqual("".join(output), 'Hello "Marcus\nnext')

    def test_stream_reader_reassembles_content_and_reports_visible_text(self) -> None:
        brain = OpenRouterBrain(api_key="unused", model="test/model")
        events = [
            {
                "model": "test/model",
                "choices": [{"delta": {"content": '{"reply":"Hel'}}],
            },
            {
                "model": "test/model",
                "choices": [
                    {
                        "delta": {"content": 'lo!","reflect":false}'},
                        "finish_reason": "stop",
                    }
                ],
            },
            {"model": "test/model", "choices": [], "usage": {"total_tokens": 12}},
        ]
        stream = [f"data: {json.dumps(event)}\n".encode() for event in events]
        stream.append(b"data: [DONE]\n")
        visible = []

        result, first_output = brain._read_stream(
            stream,
            started=time.monotonic(),
            on_text=visible.append,
            turn_id="turn-1",
        )

        self.assertEqual("".join(visible), "Hello!")
        self.assertEqual(
            result["choices"][0]["message"]["content"],
            '{"reply":"Hello!","reflect":false}',
        )
        self.assertEqual(result["usage"]["total_tokens"], 12)
        self.assertIsNotNone(first_output)


if __name__ == "__main__":
    unittest.main()
