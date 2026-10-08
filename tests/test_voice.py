import unittest
import os
import threading
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marcus.voice.tts import SpeechChunker, SpeechOutput, _plain_speech, _audio_has_frames
from marcus.observability.audit import AuditLog, read_events, session_scope


class SpeechChunkerTests(unittest.TestCase):
    def test_emits_short_opening_phrase_without_waiting_for_full_answer(self) -> None:
        chunker = SpeechChunker()

        self.assertEqual(chunker.feed("Hey! Here is the rest"), ["Hey!"])
        self.assertEqual(chunker.finish(), ["Here is the rest"])

    def test_removes_basic_markdown_before_speech(self) -> None:
        self.assertEqual(
            _plain_speech("Use **Marcus** and [the docs](https://example.com)."),
            "Use Marcus and the docs.",
        )

    def test_does_not_split_alarm_time_or_speak_orphaned_quote(self):
        chunker = SpeechChunker()
        chunks = []
        for fragment in ['You can ask your phone’s assistant: “Set an alarm for 8:', '00 AM tomorrow.', '”']:
            chunks.extend(chunker.feed(fragment))
        chunks.extend(chunker.finish())
        self.assertEqual(chunks, ['You can ask your phone’s assistant: “Set an alarm for 8:00 AM tomorrow.”'])
        self.assertEqual(_plain_speech('”'), "")

    def test_preserves_decimals_urls_abbreviations_and_closing_quotes(self):
        chunker = SpeechChunker()
        chunks = chunker.feed('Dr. Marcus uses v1.13 at example.com. “Nice!” Next sentence')
        chunks.extend(chunker.finish())
        self.assertEqual(chunks, ['Dr. Marcus uses v1.13 at example.com.', '“Nice!”', 'Next sentence'])


class FakeProcess:
    def __init__(self, release=None):
        self.release = release
        self.done = False
        self.returncode = 0

    def wait(self):
        if self.release:
            self.release.wait(2)
        self.done = True
        return self.returncode

    def poll(self):
        return self.returncode if self.done else None

    def terminate(self):
        self.returncode = -15
        self.done = True
        if self.release:
            self.release.set()


class SpeechOutputTests(unittest.TestCase):
    @patch.dict(os.environ, {"MARCUS_VOICE": "1", "MARCUS_VOICE_MODE": "stream"})
    def test_next_audio_is_prepared_while_previous_sentence_plays(self):
        with TemporaryDirectory() as directory:
            audit = AuditLog(Path(directory))
            first_playing = threading.Event()
            second_ready = threading.Event()
            release = threading.Event()
            spoken = []
            synthesized = []

            def run(arguments):
                if "-o" in arguments:
                    synthesized.append(arguments[-1])
                    Path(arguments[arguments.index("-o") + 1]).write_bytes(b"FORM" + struct.pack(">I", 24) + b"AIFFCOMM" + struct.pack(">IHI", 6, 1, 20))
                    if len(synthesized) == 2:
                        second_ready.set()
                    return FakeProcess()
                spoken.append(arguments[-1])
                if len(spoken) == 1:
                    first_playing.set()
                    return FakeProcess(release)
                return FakeProcess()

            speech = SpeechOutput(audit=audit, command="say", player="play", runner=run)
            try:
                with session_scope("voice-test"):
                    speech.feed("First complete sentence. ", turn_id="one")
                    self.assertTrue(first_playing.wait(1))
                    speech.feed("Second complete sentence. ", turn_id="one")
                    speech.finish(turn_id="one")
                self.assertTrue(second_ready.wait(1))
                self.assertEqual(len(spoken), 1)
                release.set()
                speech.drain()
                self.assertEqual(len(spoken), 2)
                events = read_events(audit.path, limit=50)
                self.assertTrue(all(item["session_id"] == "voice-test" for item in events if item["event"] == "voice.started"))
                self.assertEqual(synthesized, ["First complete sentence.", "Second complete sentence."])
            finally:
                release.set()
                speech.close()

    @patch.dict(os.environ, {"MARCUS_VOICE": "1", "MARCUS_VOICE_MODE": "buffered"})
    def test_buffered_mode_speaks_one_complete_reply_and_interrupts(self):
        arguments = []
        release = threading.Event()
        started = threading.Event()

        def run(args):
            arguments.append(args)
            started.set()
            return FakeProcess(release)

        speech = SpeechOutput(command="say", runner=run)
        try:
            speech.feed("Set an alarm for 8:")
            speech.feed('00 AM tomorrow. Then sleep well.”')
            self.assertEqual(arguments, [])
            speech.finish()
            self.assertTrue(started.wait(1))
            speech.interrupt()
            speech.drain()
            self.assertEqual(arguments[0][-1], 'Set an alarm for 8:00 AM tomorrow. Then sleep well.”')
        finally:
            speech.close()

    def test_empty_successful_audio_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.aiff"
            path.write_bytes(b"FORM" + struct.pack(">I", 24) + b"AIFFCOMM" + struct.pack(">IHI", 6, 1, 0))
            self.assertFalse(_audio_has_frames(str(path)))


if __name__ == "__main__":
    unittest.main()
