#!/usr/bin/env python3
"""One-shot box provisioner for SpecEdge.

A trimmed-down cousin of script/sweep.py: it does the one-time setup that a sweep
needs and nothing else. Given a server box and a client box (SSH targets), it:

  1. clones paths.repo into paths.remote_root on both boxes (fetch + ff-only if a
     checkout already exists) and runs `uv sync` in each;
  2. generates the client's SSH key (script/ssh_key.sh) and authorizes its public
     half on the server box -- and on the client box itself, for localhost nodes;
  3. runs script/vast_ports.sh --emit on the server box to pick a free published
     port, then prints the two values you paste into your config by hand:
         server.port  = <container port>   (SPECEDGE_PORT on the box)
         client.host  = <public_ip>:<host_port>

Nothing is written to any config file -- step 3 just prints the mapping.

Usage:
  python script/setup_server.py -f config/sweep.example.yaml
  python script/setup_server.py --server 'root@1.2.3.4 -p 22022' \
      --client 'root@5.6.7.8 -p 40000' --repo https://github.com/baguioni/specedge
  python script/setup_server.py -f config/sweep.example.yaml --skip-repo   # ports+ssh only
  python script/setup_server.py -f config/sweep.example.yaml --only-ports

Config keys read (sweep.py-compatible; CLI flags override):
  paths.repo, paths.repo_ref, paths.remote_root, paths.base_config
  server.ssh, client.ssh, ssh_identity
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "https://github.com/baguioni/specedge"
DEFAULT_REMOTE_ROOT = "/workspace/specedge"
DEFAULT_SSH_KEY = "~/.ssh/id_ed25519_server"
# Identity file used as `ssh -i <path>` for both boxes unless overridden by the
# config's `ssh_identity` or --ssh-identity.
DEFAULT_SSH_IDENTITY = "~/.ssh/vast"

# Identity file injected as `ssh -i <path>` for every remote box, unless that
# box's own ssh string already carries a -i. Set from `ssh_identity` in main().
SSH_IDENTITY: str | None = None

_T0 = time.monotonic()


def _stamp() -> str:
    return f"{time.monotonic() - _T0:7.1f}s"


def info(msg: str, *a) -> None:
    print(f"[setup {_stamp()}] {msg % a if a else msg}", flush=True)


def warn(msg: str, *a) -> None:
    print(f"[setup {_stamp()}] WARNING: {msg % a if a else msg}", file=sys.stderr, flush=True)


def die(msg: str, *a) -> None:
    print(f"[setup {_stamp()}] ERROR: {msg % a if a else msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# local / remote command execution  (lifted from script/sweep.py)
# --------------------------------------------------------------------------- #
def is_local(target: str) -> bool:
    return str(target).strip().lower() in ("", "local", "localhost")


def ssh_argv(target: str) -> list[str]:
    """`ssh` + the global identity (unless `target` already sets -i) + target args.

    `StrictHostKeyChecking=accept-new` lets a first-time box key be saved without
    a prompt (a *changed* key is still rejected) -- this is a one-shot provisioner
    run against fresh boxes.
    """
    base = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if SSH_IDENTITY and " -i " not in f" {target} ":
        base += ["-i", SSH_IDENTITY]
    return [*base, *shlex.split(str(target))]


def run(
    target: str,
    script: str,
    *,
    capture: bool = False,
    timeout: float | None = None,
    check: bool = False,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess:
    """Run a bash snippet locally (`bash -lc`) or on an SSH target."""
    if is_local(target):
        argv = ["bash", "-lc", script]
    else:
        argv = [*ssh_argv(target), script]

    kw: dict = {}
    if capture:
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.PIPE
    if input_bytes is not None:
        kw["input"] = input_bytes

    r = subprocess.run(argv, timeout=timeout, **kw)  # noqa: S603
    if check and r.returncode != 0:
        detail = ""
        if capture:
            detail = f"\n--- stdout ---\n{r.stdout.decode(errors='replace')}" \
                     f"\n--- stderr ---\n{r.stderr.decode(errors='replace')}"
        raise RuntimeError(f"command failed ({r.returncode}) on {target!r}: {script[:160]}{detail}")
    return r


def out_of(r: subprocess.CompletedProcess) -> str:
    return (r.stdout or b"").decode(errors="replace")


# --------------------------------------------------------------------------- #
# setup steps
# --------------------------------------------------------------------------- #
def root_for(cfg: dict, target: str) -> str:
    return str(REPO) if is_local(target) else cfg["remote_root"]


def _boxes(cfg: dict) -> list[tuple[str, str]]:
    """(label, ssh target) for server + client, de-duplicated, server first."""
    out: list[tuple[str, str]] = []
    for role in ("server", "client"):
        t = cfg[role]
        if t not in {x[1] for x in out}:
            out.append((role, t))
    return out


def bootstrap_repo(cfg: dict) -> None:
    """Clone <repo> into <remote_root> and run `uv sync` on each box."""
    repo = str(cfg["repo"])
    ref = str(cfg.get("repo_ref") or "").strip()

    for label, t in _boxes(cfg):
        root = root_for(cfg, t)
        if is_local(t):
            info("%s: using existing checkout at %s (skip clone)", label, root)
        else:
            co_old = f"git -C {root} checkout {shlex.quote(ref)} && " if ref else ""
            co_new = f" && git -C {root} checkout {shlex.quote(ref)}" if ref else ""
            inner = (
                f"set -e; "
                f"if [ -d {root}/.git ]; then "
                f"git -C {root} fetch --all --prune; "
                f"{co_old}git -C {root} pull --ff-only || true; "
                f"else git clone {shlex.quote(repo)} {root}{co_new}; fi"
            )
            info("%s: clone/update %s -> %s%s", label, repo, root, f" @ {ref}" if ref else "")
            run(t, f"bash -lc {shlex.quote(inner)}", check=True)

        info("%s: uv sync in %s", label, root)
        run(t, f"bash -lc {shlex.quote(f'cd {root} && uv sync')}", check=True)

    info("repo bootstrap complete")


def bootstrap_ssh(cfg: dict) -> None:
    """Generate the client's SSH key and authorize it on the server box."""
    client = cfg["client"]
    server = cfg["server"]
    croot = root_for(cfg, client)
    key_path = cfg.get("ssh_key") or DEFAULT_SSH_KEY

    if os.path.basename(key_path) != "id_ed25519_server":
        warn("ssh_key is %s but ssh_key.sh generates ~/.ssh/id_ed25519_server; "
             "make sure these refer to the same key", key_path)

    info("client: running script/ssh_key.sh (keygen is a no-op if the key exists)")
    # `|| true`: ssh_key.sh runs under `set -e` and its ssh-agent/ssh-add tail can
    # fail on a non-interactive shell *after* the key files are already written.
    run(client, f"cd {croot} && bash script/ssh_key.sh || true")

    r = run(client, f"cat {key_path}.pub", capture=True, check=True)
    pub = out_of(r).strip()
    if "\n" in pub or not pub.startswith(("ssh-", "ecdsa-", "sk-")):
        raise RuntimeError(f"unexpected public key read from client: {pub!r}")
    info("client: public key %s ... %s", pub[:24], pub[-16:])

    authorize = (
        "install -d -m700 ~/.ssh && touch ~/.ssh/authorized_keys && "
        "chmod 600 ~/.ssh/authorized_keys && "
        f"( grep -qxF {shlex.quote(pub)} ~/.ssh/authorized_keys || "
        f"printf '%s\\n' {shlex.quote(pub)} >> ~/.ssh/authorized_keys )"
    )
    info("server: authorizing key in ~/.ssh/authorized_keys")
    run(server, authorize, check=True)
    if client != server:
        info("client: self-authorizing key (for localhost nodes)")
        run(client, authorize, check=True)
    info("ssh bootstrap complete")


def discover_ports(cfg: dict) -> tuple[int, str]:
    """Run script/vast_ports.sh --emit on the server box.

    Returns (container_port, "<public_ip>:<host_port>") -- i.e. the values you
    paste into server.port and client.host of your config by hand.
    """
    target = cfg["server"]
    root = root_for(cfg, target)
    inner = f"cd {root} && bash script/vast_ports.sh --emit"
    r = run(target, f"bash -lc {shlex.quote(inner)}", capture=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"vast_ports.sh --emit failed on the server box (rc={r.returncode}): "
            f"{(r.stderr or b'').decode(errors='replace').strip()[:400]}"
        )

    kv: dict[str, str] = {}
    for line in out_of(r).splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.isupper():
            kv[k] = v.strip()

    missing = [k for k in ("PUBLIC_IP", "CONTAINER_PORT", "HOST_PORT") if not kv.get(k)]
    if missing:
        raise RuntimeError(f"vast_ports.sh --emit missing {missing}; stdout was {out_of(r)!r}")
    if not (kv["CONTAINER_PORT"].isdigit() and kv["HOST_PORT"].isdigit()):
        raise RuntimeError(f"vast_ports.sh --emit returned non-numeric ports: {kv}")

    container_port = int(kv["CONTAINER_PORT"])
    client_host = f"{kv['PUBLIC_IP']}:{kv['HOST_PORT']}"
    info("discovered ports: server binds %d, client dials %s", container_port, client_host)
    return container_port, client_host


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def load_cfg(args: argparse.Namespace) -> dict:
    """Merge the sweep-style YAML (if any) with CLI overrides into a flat dict."""
    raw: dict = {}
    if args.config:
        p = Path(args.config)
        if not p.is_absolute():
            p = REPO / p
        if not p.is_file():
            die("no such config: %s", p)
        raw = yaml.safe_load(p.read_text()) or {}

    paths = raw.get("paths") or {}
    cfg: dict = {
        "server": args.server or (raw.get("server") or {}).get("ssh"),
        "client": args.client or (raw.get("client") or {}).get("ssh"),
        "repo": args.repo or paths.get("repo") or DEFAULT_REPO,
        "repo_ref": args.repo_ref or paths.get("repo_ref") or "",
        "remote_root": args.remote_root or paths.get("remote_root") or DEFAULT_REMOTE_ROOT,
        "ssh_identity": args.ssh_identity or raw.get("ssh_identity") or DEFAULT_SSH_IDENTITY,
    }

    # ssh_key: CLI > base_config's base.ssh_key > default
    ssh_key = args.ssh_key
    if not ssh_key and paths.get("base_config"):
        bc = REPO / paths["base_config"]
        if bc.is_file():
            base_cfg = yaml.safe_load(bc.read_text()) or {}
            ssh_key = (base_cfg.get("base") or {}).get("ssh_key")
    cfg["ssh_key"] = ssh_key or DEFAULT_SSH_KEY

    if not cfg["server"]:
        die("no server ssh target (set server.ssh in the config or pass --server)")
    if not cfg["client"]:
        die("no client ssh target (set client.ssh in the config or pass --client)")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-f", "--config", help="sweep-style YAML to read server/client/paths from")
    ap.add_argument("--server", help="server ssh target, e.g. 'root@1.2.3.4 -p 22022'")
    ap.add_argument("--client", help="client ssh target")
    ap.add_argument("--repo", help=f"git URL to clone (default: {DEFAULT_REPO})")
    ap.add_argument("--repo-ref", dest="repo_ref", help="branch / tag / commit to check out")
    ap.add_argument("--remote-root", dest="remote_root",
                    help=f"checkout path on the boxes (default: {DEFAULT_REMOTE_ROOT})")
    ap.add_argument("--ssh-identity", dest="ssh_identity",
                    help=f"identity file injected as `ssh -i <path>` for both boxes "
                         f"(default: {DEFAULT_SSH_IDENTITY})")
    ap.add_argument("--ssh-key", dest="ssh_key",
                    help=f"client key path checked after keygen (default: {DEFAULT_SSH_KEY})")
    ap.add_argument("--skip-repo", action="store_true", help="skip clone + uv sync")
    ap.add_argument("--skip-ssh", action="store_true", help="skip client keygen + authorize")
    ap.add_argument("--skip-ports", action="store_true", help="skip port discovery")
    ap.add_argument("--only-ports", action="store_true",
                    help="only run port discovery (implies --skip-repo --skip-ssh)")
    args = ap.parse_args()

    cfg = load_cfg(args)

    global SSH_IDENTITY
    if cfg["ssh_identity"]:
        SSH_IDENTITY = os.path.expanduser(str(cfg["ssh_identity"]))
        if not Path(SSH_IDENTITY).is_file():
            warn("ssh_identity %s not found on this machine", SSH_IDENTITY)

    do_repo = not (args.skip_repo or args.only_ports)
    do_ssh = not (args.skip_ssh or args.only_ports)
    do_ports = not args.skip_ports

    info("server -> %s", cfg["server"])
    info("client -> %s", cfg["client"])
    info("repo   -> %s%s into %s",
         cfg["repo"], f" @ {cfg['repo_ref']}" if cfg["repo_ref"] else "", cfg["remote_root"])

    if do_repo:
        try:
            bootstrap_repo(cfg)
        except RuntimeError as e:
            die("%s", e)

    if do_ssh:
        try:
            bootstrap_ssh(cfg)
        except RuntimeError as e:
            die("%s", e)

    mapping: tuple[int, str] | None = None
    if do_ports:
        try:
            mapping = discover_ports(cfg)
        except RuntimeError as e:
            die("%s", e)

    print("\n==== setup complete ====")
    print(f"  repo bootstrap : {'done' if do_repo else 'skipped'}")
    print(f"  ssh bootstrap  : {'done' if do_ssh else 'skipped'}")
    if mapping:
        port, host = mapping
        print("\n  add these to your config by hand:")
        print(f"    server:\n      port: {port}")
        print(f"    client:\n      host: \"{host}\"")
    elif do_ports:
        print("  port discovery : failed")
    else:
        print("  port discovery : skipped")


if __name__ == "__main__":
    main()
