import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from importlib import reload

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="dragoon_memory_")
        self.original_data_dir = os.environ.get("DRAGOON_DATA_DIR")
        os.environ["DRAGOON_DATA_DIR"] = self.temp_dir
        import core.memory as memory
        reload(memory)
        self.memory = memory

    def tearDown(self):
        if self.original_data_dir is None:
            os.environ.pop("DRAGOON_DATA_DIR", None)
        else:
            os.environ["DRAGOON_DATA_DIR"] = self.original_data_dir

    def test_update_context_and_get_context(self):
        self.memory.update_context("favorite_color", "blue")
        self.memory.update_context("last_command", "Set a reminder for 5pm")

        context = self.memory.get_context(n=5)

        self.assertIn("context", context)
        self.assertEqual(context["context"]["favorite_color"], "blue")
        self.assertEqual(context["context"]["last_command"], "Set a reminder for 5pm")
        self.assertIsInstance(context["commands"], list)

    def test_get_context_excludes_stale_context_rows(self):
        stale_time = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        with self.memory._get_connection() as conn:
            conn.execute(
                "INSERT INTO context(key, value, updated_at) VALUES (?, ?, ?)",
                ("stale_fact", '"old value"', stale_time),
            )
            conn.commit()

        self.memory.update_context("fresh_fact", "new value")

        context = self.memory.get_context(n=5)
        self.assertNotIn("stale_fact", context["context"])
        self.assertEqual(context["context"]["fresh_fact"], "new value")


if __name__ == "__main__":
    unittest.main()
