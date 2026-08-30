"""A clone carries no generated dataset, so the API must start without one.

`artifacts/data` is regenerable and therefore gitignored. Anyone who clones the
repo and runs uvicorn before `generate_data.py` used to get a bare
FileNotFoundError traceback and a dead server; the views that read committed
reports were unreachable too, even though they need no dataset at all.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_without_data(monkeypatch):
    from predictops.api import app as app_module

    def _missing():
        raise FileNotFoundError("artifacts/data/telemetry.parquet")

    monkeypatch.setattr(app_module, "prepare", _missing)
    with TestClient(app_module.app) as c:
        yield c


def test_api_starts_without_a_dataset(client_without_data):
    r = client_without_data.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "setup_required"
    assert body["data_loaded"] is False
    assert body["model_loaded"] is False


def test_health_names_the_command_that_fixes_it(client_without_data):
    err = client_without_data.get("/api/health").json()["setup_error"]
    assert err and "generate_data.py" in err


@pytest.mark.parametrize("route", ["/api/fleet/overview", "/api/system"])
def test_data_backed_routes_return_503_with_the_hint(client_without_data, route):
    r = client_without_data.get(route)
    assert r.status_code == 503, route
    assert "generate_data.py" in r.json()["detail"]


@pytest.mark.parametrize("route", [
    "/api/experiments", "/api/trajectories", "/api/changelog",
])
def test_registry_views_still_work_without_a_dataset(client_without_data, route):
    """These read the registry and committed reports, not telemetry."""
    assert client_without_data.get(route).status_code == 200, route
