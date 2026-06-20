from importlib.metadata import version

import ope


def test_version_exists() -> None:
    assert isinstance(ope.__version__, str)


def test_version_matches_installed_metadata() -> None:
    # fails if the package was not installed, or if the version source drifted
    assert ope.__version__ == version("options-pricing-engine")
