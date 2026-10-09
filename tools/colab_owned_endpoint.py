"""Stop only history-proven owned Colab endpoints; local session names are not server identity."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


COLAB_PYTHON = Path.home() / ".local/share/uv/tools/google-colab-cli/bin/python"


def owned_endpoint(session, history, owned_sessions):
    """Derive exactly one endpoint from this never-reused owned session's creation record."""
    history = Path(history)
    owned = json.loads(Path(owned_sessions).read_text())
    if not isinstance(owned, list) or session not in owned:
        raise ValueError("session is not in the study ownership registry")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", session) or history.name != session + ".jsonl":
        raise ValueError("history does not belong to this session")
    raw = history.read_text()
    if raw and not raw.endswith("\n"):
        raise ValueError("partial ownership history; retry only after complete record")
    created = [
        record for line in raw.splitlines() if (record := json.loads(line)).get("event_type") == "session_created"
    ]
    if len(created) != 1:
        raise ValueError("missing or ambiguous session creation history")
    endpoint = created[0].get("endpoint")
    if not isinstance(endpoint, str) or not re.fullmatch(r"[a-zA-Z0-9._-]+", endpoint):
        raise ValueError("invalid owned endpoint")
    return endpoint


def prove_owned(session, endpoint, history, owned_sessions):
    if owned_endpoint(session, history, owned_sessions) != endpoint:
        raise ValueError("endpoint does not match owned session history")
    created = next(
        json.loads(line)
        for line in Path(history).read_text().splitlines()
        if json.loads(line).get("event_type") == "session_created"
    )
    return {
        "session": session,
        "endpoint": endpoint,
        "history": str(Path(history).resolve()),
        "created_record_sha256": hashlib.sha256(json.dumps(created, sort_keys=True).encode()).hexdigest(),
    }


def list_endpoints(client):
    """Direct server API: never use sync_sessions, which can swallow an auth SystemExit."""
    return sorted({assignment.endpoint for assignment in client.list_assignments()})


def stop_owned(client, *, session, endpoint, history, owned_sessions):
    proof = prove_owned(session, endpoint, history, owned_sessions)
    before = list_endpoints(client)
    if endpoint not in before:
        return {**proof, "absent": True, "already_absent": True, "endpoints": before}
    try:
        client.unassign(endpoint)
    except Exception:
        # An ambiguous mutation is successful only if a fresh server query proves absence.
        after = list_endpoints(client)
        if endpoint in after:
            raise
        return {**proof, "absent": True, "already_absent": False, "endpoints": after}
    after = list_endpoints(client)
    if endpoint in after:
        raise RuntimeError("owned endpoint termination not yet confirmed")
    return {**proof, "absent": True, "already_absent": False, "endpoints": after}


def endpoint_call(
    operation, *, session=None, endpoint=None, history=None, owned_sessions=None, colab_python=COLAB_PYTHON, timeout=90
):
    """Use the installed CLI environment from a project interpreter without importing its dependencies."""
    if operation not in ("list", "stop"):
        raise ValueError("unsupported endpoint operation")
    args = [str(colab_python), str(Path(__file__).resolve()), operation, "--direct"]
    if operation == "stop":
        prove_owned(session, endpoint, history, owned_sessions)
        for key, value in (
            ("session", session),
            ("endpoint", endpoint),
            ("history", history),
            ("owned-sessions", owned_sessions),
        ):
            args.extend(["--" + key, str(value)])
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)
    value = json.loads(result.stdout)
    if not isinstance(value.get("endpoints"), list) or (operation == "stop" and value.get("absent") is not True):
        raise ValueError("endpoint API acknowledgement malformed")
    return value


def authenticated_client():
    """Reuse existing OAuth credentials noninteractively, without modifying account or local session state."""
    from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH
    from colab_cli.client import Client, Prod
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2.credentials import Credentials

    credentials = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
    if not credentials.valid:
        if not credentials.refresh_token:
            raise RuntimeError("existing Colab credentials cannot be refreshed noninteractively")
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError("Colab authentication failed")
    return Client(Prod(), AuthorizedSession(credentials))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("list", "stop"))
    parser.add_argument("--session")
    parser.add_argument("--endpoint")
    parser.add_argument("--history", type=Path)
    parser.add_argument("--owned-sessions", type=Path)
    parser.add_argument("--colab-python", type=Path, default=COLAB_PYTHON)
    parser.add_argument("--direct", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    kwargs = {
        "session": args.session,
        "endpoint": args.endpoint,
        "history": args.history,
        "owned_sessions": args.owned_sessions,
    }
    if args.operation == "stop" and any(v is None for v in kwargs.values()):
        parser.error("stop requires --session, --endpoint, --history and --owned-sessions")
    if args.direct:
        if args.operation == "stop":
            prove_owned(**kwargs)
        client = authenticated_client()
        result = {"endpoints": list_endpoints(client)} if args.operation == "list" else stop_owned(client, **kwargs)
    else:
        result = endpoint_call(args.operation, **kwargs, colab_python=args.colab_python)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
