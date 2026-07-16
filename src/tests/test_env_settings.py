import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from utils import read_env_bool
import model_config


class EnvironmentSettingsTest(unittest.TestCase):
    def test_model_rate_limiter_defaults_to_chatopenai_behavior(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            model_config,
            "_MODEL_RATE_LIMITER",
            None,
        ):
            self.assertIsNone(model_config._get_model_rate_limiter())
            self.assertNotIn("rate_limiter", model_config._transport_options())

    def test_missing_variable_is_an_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(KeyError):
                read_env_bool("ENABLE_THINKING")

    def test_parses_true_and_false(self) -> None:
        with patch.dict(os.environ, {"ENABLE_THINKING": "true"}):
            self.assertTrue(read_env_bool("ENABLE_THINKING"))
        with patch.dict(os.environ, {"ENABLE_THINKING": "false"}):
            self.assertFalse(read_env_bool("ENABLE_THINKING"))

    def test_rejects_unknown_boolean_value(self) -> None:
        with patch.dict(
            os.environ,
            {"ENABLE_THINKING": "sometimes"},
        ):
            with self.assertRaisesRegex(ValueError, "ENABLE_THINKING"):
                read_env_bool("ENABLE_THINKING")

if __name__ == "__main__":
    unittest.main()
