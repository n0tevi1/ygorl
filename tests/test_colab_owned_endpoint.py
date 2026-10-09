"""Server endpoint identity survives local CLI name loss; ambiguous termination fails closed."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "colab_owned_endpoint", Path(__file__).parents[1] / "tools/colab_owned_endpoint.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def proof(tmp_path):
    history = tmp_path / "owned-vm0.jsonl"
    history.write_text(
        json.dumps({"event_type": "session_created", "endpoint": "gpu-owned", "timestamp": "frozen"}) + "\n"
    )
    registry = tmp_path / "owned-sessions.json"
    registry.write_text(json.dumps(["owned-vm0"]))
    return {"session": "owned-vm0", "endpoint": "gpu-owned", "history": history, "owned_sessions": registry}


class Client:
    def __init__(self, endpoints):
        self.endpoints = set(endpoints)
        self.stopped = []

    def list_assignments(self):
        return [SimpleNamespace(endpoint=value) for value in self.endpoints]

    def unassign(self, endpoint):
        self.stopped.append(endpoint)
        self.endpoints.remove(endpoint)


def test_lost_local_name_still_stops_only_proven_owned_endpoint(proof):
    client = Client(["gpu-owned", "unrelated-user-machine"])
    result = module.stop_owned(client, **proof)
    assert result["absent"] is True
    assert result["endpoints"] == ["unrelated-user-machine"]
    assert client.stopped == ["gpu-owned"]


def test_unknown_endpoint_or_session_never_mutates_server(proof):
    client = Client(["gpu-owned", "unrelated"])
    with pytest.raises(ValueError, match="does not match"):
        module.stop_owned(client, **{**proof, "endpoint": "unrelated"})
    with pytest.raises(ValueError, match="ownership registry"):
        module.stop_owned(client, **{**proof, "session": "not-owned"})
    assert client.stopped == []


def test_absence_is_confirmed_on_server_not_inferred_from_local_state(proof):
    client = Client(["unrelated"])
    result = module.stop_owned(client, **proof)
    assert result["already_absent"] is True
    assert client.stopped == []


def test_auth_failure_cannot_be_interpreted_as_no_assignments(proof):
    class FailedAuth(Client):
        def list_assignments(self):
            raise SystemExit(1)

    with pytest.raises(SystemExit):
        module.stop_owned(FailedAuth([]), **proof)


def test_successful_unassign_reply_is_not_enough_without_server_absence(proof):
    client = Client(["gpu-owned"])
    client.unassign = lambda endpoint: None
    with pytest.raises(RuntimeError, match="not yet confirmed"):
        module.stop_owned(client, **proof)


def test_ambiguous_unassign_accepts_only_fresh_confirmed_absence(proof):
    client = Client(["gpu-owned"])

    def timed_out(endpoint):
        client.endpoints.remove(endpoint)
        raise TimeoutError("response lost")

    client.unassign = timed_out
    assert module.stop_owned(client, **proof)["absent"] is True
    client.endpoints.add("gpu-owned")
    client.unassign = lambda endpoint: (_ for _ in ()).throw(TimeoutError("not completed"))
    with pytest.raises(TimeoutError):
        module.stop_owned(client, **proof)


def test_reused_session_name_or_partial_history_is_rejected(proof):
    path = proof["history"]
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="ambiguous"):
        module.prove_owned(**proof)
    path.write_text('{"event_type":')
    with pytest.raises(ValueError, match="partial"):
        module.prove_owned(**proof)


def test_authenticated_client_receives_request_capable_authorized_session(monkeypatch):
    credentials = SimpleNamespace(valid=True)
    requests = []

    class AuthorizedSession:
        def __init__(self, value):
            assert value is credentials

        def request(self, method, url):
            requests.append((method, url))
            return []

    class ApiClient:
        def __init__(self, environment, session):
            self.session = session

        def list_assignments(self):
            return self.session.request("GET", "assignments")

    for name, fake in {
        "colab_cli.auth": SimpleNamespace(PUBLIC_SCOPES=["scope"], TOKEN_CONFIG_PATH="existing-token"),
        "colab_cli.client": SimpleNamespace(Client=ApiClient, Prod=lambda: None),
        "google.auth.transport.requests": SimpleNamespace(AuthorizedSession=AuthorizedSession, Request=lambda: None),
        "google.oauth2.credentials": SimpleNamespace(
            Credentials=SimpleNamespace(from_authorized_user_file=lambda *args: credentials)
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, fake)
    assert module.list_endpoints(module.authenticated_client()) == []
    assert requests == [("GET", "assignments")]
