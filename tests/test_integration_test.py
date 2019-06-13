import random

import numpy as np

from examples.demo import Demo

# ensure that every demo run behave constantly equal
random.seed(1)
np.random.seed(1)


def test_flatland_000():
    Demo.run_example_flatland_000()
    # TODO test assertions


def test_flatland_001():
    Demo.run_example_flatland_001()
    # TODO test assertions


def test_network_000():
    Demo.run_example_network_000()
    # TODO test assertions


def test_network_001():
    Demo.run_example_network_001()
    # TODO test assertions


def test_network_002():
    Demo.run_example_network_002()
    # TODO test assertions


def test_complex_scene():
    Demo.run_complex_scene()
    # TODO test assertions


def test_generate_complex_scenario():
    Demo.run_generate_complex_scenario()
    # TODO test assertions


def test_generate_random_scenario():
    Demo.run_generate_random_scenario()
    # TODO test assertions
