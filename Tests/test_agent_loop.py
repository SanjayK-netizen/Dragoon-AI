import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.agent_loop import run_agent_loop


class AgentLoopTests(unittest.TestCase):
    def test_calculation_tool_executes(self):
        result = run_agent_loop("Calculate 12 + 7")
        self.assertIn("19", result)

    def test_retry_once_before_failing(self):
        calls = {"count": 0}

        def flaky_tool(value):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("temporary failure")
            return {"ok": True, "value": value}

        result = run_agent_loop("Use flaky_tool with value=42", tool_registry={"flaky_tool": flaky_tool})
        self.assertIn("42", result)
        self.assertEqual(calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
