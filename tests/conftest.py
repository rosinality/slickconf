from pathlib import Path

import pytest


@pytest.fixture
def assets_dir(request: pytest.FixtureRequest) -> Path:
    return request.node.path.parent.with_suffix("") / "assets"
