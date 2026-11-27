"""Online client: a reconnect that fails is tried again (the client must not stay offline)."""
import queue
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "template"), str(ROOT)]

from online import client as cl          # noqa: E402


class Sock:
    def __init__(self, up):
        self.connected = up

    def disconnect(self):
        self.connected = False

    def call(self, *a, **kw):
        if not self.connected:
            raise RuntimeError("/ is not a connected namespace.")
        return {}


class Reconnect(unittest.TestCase):
    def setUp(self):
        self.sleep, cl.time.sleep = cl.time.sleep, lambda s: None

    def tearDown(self):
        cl.time.sleep = self.sleep

    def test_a_failed_reconnect_is_tried_again(self):
        acc = SimpleNamespace(check_login=lambda: False)
        c = cl.Conn(SimpleNamespace(acc=acc, brain=None), "test")
        tries = []

        def connect(tries_=4):
            tries.append(1)
            c.sio, c.closing = Sock(len(tries) > 1), False     # the first reconnect fails
            if len(tries) == 1:
                raise ConnectionError("server busy")
        c.connect = connect
        c.sio = Sock(False)
        c.events.put(("disconnect", c.sio))
        c.next_event()                   # reconnect fails, the retry is queued
        c.next_event()                   # ... and taken: the second reconnect succeeds
        self.assertEqual(len(tries), 2)
        self.assertTrue(c.sio.connected)
        # a socket that dropped without an event: the next call queues the reconnect
        c.sio.connected = False
        with self.assertRaises(RuntimeError):
            c.call("x", {})
        c.next_event()
        self.assertEqual(len(tries), 3)
        self.assertTrue(c.sio.connected)
        with self.assertRaises(queue.Empty):
            c.events.get_nowait()


if __name__ == "__main__":
    unittest.main()
