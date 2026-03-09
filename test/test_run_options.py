"""Tests for ortpy.RunOptions."""
import pytest

import ortpy as ort


class TestRunOptionsBasic:
    def test_default_construction(self):
        opts = ort.RunOptions()
        assert opts is not None


class TestRunOptionsLogVerbosity:
    def test_get_set_verbosity(self):
        opts = ort.RunOptions()
        opts.run_log_verbosity_level = 5
        assert opts.run_log_verbosity_level == 5

    def test_default_verbosity(self):
        opts = ort.RunOptions()
        # Default is 0
        assert opts.run_log_verbosity_level == 0


class TestRunOptionsLogSeverity:
    def test_get_set_severity(self):
        opts = ort.RunOptions()
        opts.run_log_severity_level = 3
        assert opts.run_log_severity_level == 3


class TestRunOptionsRunTag:
    def test_get_set_tag(self):
        opts = ort.RunOptions()
        opts.run_tag = "test_tag"
        assert opts.run_tag == "test_tag"

    def test_default_tag(self):
        opts = ort.RunOptions()
        assert opts.run_tag == ""


class TestRunOptionsTerminate:
    def test_set_unset_terminate(self):
        opts = ort.RunOptions()
        opts.set_terminate()
        opts.unset_terminate()


class TestRunOptionsConfig:
    def test_add_and_get_config_entry(self):
        opts = ort.RunOptions()
        opts.add_run_config_entry("test.key", "test_value")
        result = opts.get_run_config_entry("test.key")
        assert result == "test_value"

    def test_get_nonexistent_key(self):
        opts = ort.RunOptions()
        result = opts.get_run_config_entry("nonexistent")
        assert result is None


class TestRunOptionsWithInference:
    def test_run_with_options(self, add_session):
        import numpy as np
        opts = ort.RunOptions()
        opts.run_tag = "inference_test"
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = add_session.run({"A": a, "B": b}, run_options=opts)
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_run_with_terminate_unset(self, add_session):
        import numpy as np
        opts = ort.RunOptions()
        opts.set_terminate()
        opts.unset_terminate()
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = add_session.run({"A": a, "B": b}, run_options=opts)
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_run_without_options(self, add_session):
        import numpy as np
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = add_session.run({"A": a, "B": b})
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])
