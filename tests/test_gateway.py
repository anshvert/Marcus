import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from marcus.runtime.gateway import make_gateway
from marcus.runtime.client import GatewayClient
from test_runtime import fake_runtime, FakeBrain


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = fake_runtime(self.root, FakeBrain())
        self.server = make_gateway(self.runtime, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.token = (self.root / "gateway-token").read_text()

    def tearDown(self):
        self.server.RequestHandlerClass.stopping.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.runtime.close()
        self.temp.cleanup()

    def post(self, *, token=None, origin=None, host=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        if origin:
            headers["Origin"] = origin
        if host:
            headers["Host"] = host
        request = Request(self.url + "/api/turns", data=json.dumps({"text": "hi"}).encode(), headers=headers)
        return urlopen(request, timeout=2)

    def test_requires_token_and_rejects_foreign_origins_and_hostnames(self):
        for options, status in [({}, 401), ({"token": self.token, "origin": "https://evil.example"}, 403),
                                ({"token": self.token, "host": "evil.example"}, 403)]:
            with self.assertRaises(HTTPError) as raised:
                self.post(**options)
            self.assertEqual(raised.exception.code, status)

    def test_client_receives_streamed_reply_and_persisted_job(self):
        client = GatewayClient(self.url, token_path=self.root / "gateway-token")
        visible = []
        result = client.turn("hello", session_id="test", on_text=visible.append)
        self.assertEqual(result["reply"], "Reply: hello")
        self.assertEqual("".join(visible), "Reply: hello")
        self.assertEqual(client.request("/api/jobs")["jobs"][0]["status"], "completed")

    def test_rejects_remote_bind(self):
        with self.assertRaises(ValueError):
            make_gateway(self.runtime, host="0.0.0.0")
