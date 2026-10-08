import unittest

from marcus.observability.bus import EventBus


class EventBusTests(unittest.TestCase):
    def test_slow_subscriber_is_bounded_and_replay_has_no_subscription_gap(self):
        bus = EventBus(capacity=2)
        bus.publish({"event": "first"})
        with bus.subscribe(replay=2) as events:
            bus.publish({"event": "second"})
            bus.publish({"event": "third"})
            self.assertEqual(events.qsize(), 2)
            self.assertEqual(events.get_nowait()["event"], "second")
            self.assertEqual(events.get_nowait()["event"], "third")

    def test_events_are_snapshots_not_mutable_shared_state(self):
        bus = EventBus()
        record = {"event": "test", "data": {"text": "original"}}
        bus.publish(record)
        record["data"]["text"] = "changed"
        with bus.subscribe(replay=1) as events:
            self.assertEqual(events.get_nowait()["data"]["text"], "original")
