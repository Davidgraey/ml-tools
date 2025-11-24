'''

https://docs.pytest.org/en/6.2.x/fixture.html

'''
import pytest

@pytest.fixture()
def base_fixture():
    return True
