import os
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
