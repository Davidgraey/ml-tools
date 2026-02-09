"""

https://docs.pytest.org/en/6.2.x/fixture.html

"""

from ml_tools.generators import RandomDatasetGenerator
import pytest


@pytest.fixture()
def base_fixture():
    return True


@pytest.fixture()
def regression_dataset():
    r = RandomDatasetGenerator(random_seed=42)
    return r.generate(
        task="regression", num_samples=1500, num_features=3, noise_scale=1.5
    )
