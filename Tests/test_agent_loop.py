import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.agent_loop import AgentState, VALID_STATES, run_agent_loop
from Tools.register import REGISTRY


class AgentLoopTests(unittest.TestCase):
    def test_states_are_explicit_and_registry_has_safe_tools(self):
        self.assertEqual(VALID_STATES, [state.value for state in AgentState])
        self.assertTrue({"calculate", "get_time", "set_reminder", "open_local_file"}.issubset(REGISTRY))

    def test_calculation_tool_executes(self):
        result = run_agent_loop("Calculate 12 + 7")
        self.assertIn("19", result)

    def test_natural_language_safe_tools_execute(self):
        self.assertIn("1081", run_agent_loop("Calculate 47 times 23"))
        with patch.dict(os.environ, {"DRAGOON_DATA_DIR": tempfile.mkdtemp(prefix="dragoon_agent_")}):
            reminder = run_agent_loop("Set a reminder to take medicine at 9am")
        self.assertIn("take medicine", reminder)
        self.assertIn("9am", reminder)

    def test_invalid_builtin_arguments_are_rejected(self):
        result = run_agent_loop('Use {"tool": "calculate", "args": {"unexpected": 1}}')
        self.assertIn("couldn't safely execute", result)

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

    def test_retry_cap_is_two_attempts(self):
        calls = {"count": 0}

        def broken_tool():
            calls["count"] += 1
            raise RuntimeError("permanent failure")

        result = run_agent_loop(
            'Use {"tool": "broken_tool", "args": {}}',
            tool_registry={"broken_tool": broken_tool},
        )
        self.assertIn("tool error", result)
        self.assertEqual(calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
