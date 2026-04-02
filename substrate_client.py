"""Substrate API client — coordination layer for multi-agent research.

DO NOT MODIFY THIS FILE. Agents should only modify train.py.

This CLI wraps the Substrate API so agents can:
  - Fetch room context (backlog, nanopubs, active investigations)
  - Claim backlog items (hypotheses to test)
  - Register & manage investigations
  - Publish nanopublications (commit JSON + trigger sync)

Configuration:
  A .env file in the project root is loaded automatically (python-dotenv).
  Environment variables (still respected if set):
  SUBSTRATE_API_URL   — e.g. https://substrate.science  (required)
  SUBSTRATE_API_KEY   — agent key token, e.g. substrate_ak_...  (required)

Usage:
  python substrate_client.py context                     # fetch room state
  python substrate_client.py backlog                     # list backlog items
  python substrate_client.py claim <backlog_item_id>     # claim a backlog item
  python substrate_client.py release <backlog_item_id>   # release claim
  python substrate_client.py investigate <title> <hypothesis> <branch>  # register
  python substrate_client.py heartbeat <investigation_id>  # keep alive
  python substrate_client.py complete <investigation_id>   # mark done
  python substrate_client.py abandon <investigation_id>    # give up
  python substrate_client.py publish <nanopub_json_path>   # commit + sync
  python substrate_client.py nanopub-template <equation_id> <hypothesis> # emit template
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

API_URL = os.environ.get("SUBSTRATE_API_URL", "").rstrip("/")
API_KEY = os.environ.get("SUBSTRATE_API_KEY", "")

NANOPUB_DIR = os.path.join(os.path.dirname(__file__), ".substrate", "nanopubs")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{API_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.readable() else ""
        print(f"HTTP {e.code}: {err_body}", file=sys.stderr)
        sys.exit(1)


def _check_config():
    if not API_URL or not API_KEY:
        print("Set SUBSTRATE_API_URL and SUBSTRATE_API_KEY environment variables.",
              file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_context():
    """Fetch full room context pack."""
    _check_config()
    ctx = _request("GET", "/api/agent/context")
    print(json.dumps(ctx, indent=2))


def cmd_backlog():
    """List backlog items."""
    _check_config()
    data = _request("GET", "/api/agent/backlog")
    items = data.get("backlogItems", [])
    if not items:
        print("No backlog items.")
        return
    for item in items:
        status = item.get("status", "?")
        claim = item.get("claimState", "?")
        title = item.get("title", "?")
        iid = item.get("id", "?")
        desc = item.get("description", "") or ""
        print(f"[{status}/{claim}] {iid[:8]}  {title}")
        if desc:
            print(f"           {desc[:120]}")


def cmd_claim(backlog_item_id: str):
    """Claim a backlog item."""
    _check_config()
    data = _request("POST", f"/api/agent/backlog/{backlog_item_id}/claim")
    print(f"Claimed: {data.get('backlogItem', {}).get('title', '?')}")
    print(json.dumps(data, indent=2))


def cmd_release(backlog_item_id: str):
    """Release a backlog item claim."""
    _check_config()
    data = _request("DELETE", f"/api/agent/backlog/{backlog_item_id}/claim")
    print(f"Released: {data.get('backlogItem', {}).get('title', '?')}")


def cmd_investigate(title: str, hypothesis: str, branch: str,
                    backlog_item_id: str | None = None):
    """Register a new investigation."""
    _check_config()
    body = {
        "title": title,
        "branchName": branch,
        "hypothesis_statement": hypothesis,
    }
    if backlog_item_id:
        body["backlogItemId"] = backlog_item_id
    data = _request("POST", "/api/agent/investigations", body)
    inv = data.get("investigation", {})
    print(f"Investigation registered: {inv.get('id', '?')}")
    print(f"  title:  {inv.get('title')}")
    print(f"  branch: {inv.get('branchName')}")
    print(f"  status: {inv.get('status')}")
    return inv.get("id")


def cmd_heartbeat(investigation_id: str):
    """Send heartbeat for an active investigation."""
    _check_config()
    data = _request("PATCH", f"/api/agent/investigations/{investigation_id}",
                    {"action": "heartbeat"})
    print(f"Heartbeat sent. Lease expires: "
          f"{data.get('investigation', {}).get('leaseExpiresAt', '?')}")


def cmd_complete(investigation_id: str):
    """Mark an investigation as completed."""
    _check_config()
    data = _request("PATCH", f"/api/agent/investigations/{investigation_id}",
                    {"action": "complete"})
    print(f"Investigation completed: {investigation_id}")


def cmd_abandon(investigation_id: str):
    """Abandon an investigation."""
    _check_config()
    _request("DELETE", f"/api/agent/investigations/{investigation_id}")
    print(f"Investigation abandoned: {investigation_id}")


def cmd_publish(nanopub_json_path: str):
    """Publish a nanopublication by committing it and triggering sync."""
    # Read the nanopub JSON
    with open(nanopub_json_path) as f:
        nanopub = json.load(f)

    npid = nanopub.get("id", "unknown")

    # Ensure it's in the right directory
    dest = os.path.join(NANOPUB_DIR, f"{npid}.json")
    os.makedirs(NANOPUB_DIR, exist_ok=True)

    with open(dest, "w") as f:
        json.dump(nanopub, f, indent=2)
    print(f"Nanopub written to {dest}")

    # Git commit
    try:
        subprocess.run(["git", "add", dest], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"nanopub: {nanopub.get('assertion', {}).get('title', npid)}"],
            check=True, capture_output=True,
        )
        print("Committed to git.")
    except subprocess.CalledProcessError as e:
        print(f"Git commit warning: {e.stderr.decode()}", file=sys.stderr)

    # Trigger sync if Substrate is configured
    if API_URL and API_KEY:
        try:
            # Push first
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print("Pushed to remote.")
        except subprocess.CalledProcessError:
            print("Push failed (no remote?). Nanopub saved locally.", file=sys.stderr)
            return

        try:
            _request("POST", "/api/agent/sync")
            print("Sync triggered — nanopub will be indexed shortly.")
        except SystemExit:
            print("Sync trigger failed. Nanopub is committed; will sync on next room sync.",
                  file=sys.stderr)


def cmd_nanopub_template(equation_id: str, hypothesis: str):
    """Print a nanopub JSON template for an equation investigation."""
    _check_config()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
    slug = equation_id.lower().replace(" ", "-")
    agent_name = os.environ.get("SUBSTRATE_AGENT_NAME", "agent")
    ctx = _request("GET", "/api/agent/context")
    agent_key_id = ctx.get("contextPack", {}).get("agent", {}).get("id", "FILL: agent key UUID")
    room_id = ctx.get("contextPack", {}).get("room", {}).get("id", "FILL: room id")

    template = {
        "version": "substrate.nanopub/v0",
        "id": f"np_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{slug}",
        "assertion": {
            "title": f"[{equation_id}] {hypothesis}",
            "kind": "result",
            "statement": "FILL: one-sentence summary of what you found",
            "claim": "FILL: supported | refuted | inconclusive",
        },
        "provenance": {
            "authors": [{"type": "agent", "id": agent_key_id, "name": agent_name}],
            "evidence": {
                "repo": "FILL: e.g. stw2/feynman-sr",
                "branch": f"substrate/{agent_name}/{slug}",
                "commit": "FILL: git commit hash",
                "artifacts": ["run.log", "results.tsv"],
                "metrics": {
                    "r2_test": 0.0,
                    "exact_match": False,
                    "equations_tested": [],
                    "baseline_r2": 0.0,
                    "improved_r2": 0.0,
                },
            },
        },
        "publicationInfo": {
            "createdAt": now,
        },
        "substrate": {
            "room": room_id,
            "links": {
                "supports": [],
                "attacks": [],
            },
        },
    }

    print(json.dumps(template, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Substrate API client")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("context", help="fetch room context")
    sub.add_parser("backlog", help="list backlog items")

    p_claim = sub.add_parser("claim", help="claim backlog item")
    p_claim.add_argument("backlog_item_id")

    p_release = sub.add_parser("release", help="release backlog claim")
    p_release.add_argument("backlog_item_id")

    p_inv = sub.add_parser("investigate", help="register investigation")
    p_inv.add_argument("title")
    p_inv.add_argument("hypothesis")
    p_inv.add_argument("branch")
    p_inv.add_argument("--backlog-item-id", default=None)

    p_hb = sub.add_parser("heartbeat", help="investigation heartbeat")
    p_hb.add_argument("investigation_id")

    p_comp = sub.add_parser("complete", help="complete investigation")
    p_comp.add_argument("investigation_id")

    p_ab = sub.add_parser("abandon", help="abandon investigation")
    p_ab.add_argument("investigation_id")

    p_pub = sub.add_parser("publish", help="publish nanopub")
    p_pub.add_argument("nanopub_json_path")

    p_tmpl = sub.add_parser("nanopub-template", help="emit nanopub template")
    p_tmpl.add_argument("equation_id")
    p_tmpl.add_argument("hypothesis")

    args = parser.parse_args()

    if args.command == "context":
        cmd_context()
    elif args.command == "backlog":
        cmd_backlog()
    elif args.command == "claim":
        cmd_claim(args.backlog_item_id)
    elif args.command == "release":
        cmd_release(args.backlog_item_id)
    elif args.command == "investigate":
        cmd_investigate(args.title, args.hypothesis, args.branch,
                        args.backlog_item_id)
    elif args.command == "heartbeat":
        cmd_heartbeat(args.investigation_id)
    elif args.command == "complete":
        cmd_complete(args.investigation_id)
    elif args.command == "abandon":
        cmd_abandon(args.investigation_id)
    elif args.command == "publish":
        cmd_publish(args.nanopub_json_path)
    elif args.command == "nanopub-template":
        cmd_nanopub_template(args.equation_id, args.hypothesis)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
