from __future__ import annotations

import pytest

from harness.config import load_env, load_experiment, load_profile


@pytest.fixture
def exp_full():
    return load_experiment(load_profile("full"))


@pytest.fixture
def exp_smoke():
    return load_experiment(load_profile("smoke"))


@pytest.fixture
def env_main():
    return load_env("main")


@pytest.fixture
def env_test():
    return load_env("test")
