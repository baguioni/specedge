#!/usr/bin/env python3
"""Benchmark sweep orchestrator for SpecEdge.

Drives a matrix of experiments across a server box and a client box (each either
"local" or an SSH target). For every experiment it:

  1. renders a config = base_config <- common <- experiment.overrides, with
     base.exp_name set to the experiment name and base.result_path forced from
     the sweep config; server.port / client.host come from port discovery on the
     server box (or the sweep config, with --no-discover-ports);
  2. pushes the rendered config to both boxes (and keeps a local copy);
  3. starts src/script/batch_server.py on the server box, detached, and waits for
     the readiness marker in its stdout/log;
  4. runs src/script/client_host.py on the client box and waits for it to finish;
  5. lets the server shut itself down -- clients send the Done RPC when finished,
     which trips the server's shutdown once every client has checked in -- with
     SIGINT then SIGKILL as fallbacks;
  6. pulls result/<exp_name>/ from both boxes into collect_to/<exp_name>/.

A run's server process is always torn down in a finally block, so a failure or
Ctrl+C does not leak a GPU process on the server box.

Usage:
  python script/sweep.py -f config/sweep.example.yaml
  python script/sweep.py -f config/sweep.example.yaml --list
  python script/sweep.py -f config/sweep.example.yaml --bootstrap-repo --bootstrap-ssh --dry-run
  python script/sweep.py -f config/sweep.example.yaml --only saguaro_b12_geom,proactive_bl3
  python script/sweep.py -f config/sweep.example.yaml --from proactive_bl4
  python script/sweep.py -f config/sweep.example.yaml --dry-run      # render merged configs only

--dry-run renders the fully merged per-experiment configs to paths.rendered_dir
and exits without touching any box (no port discovery, no server/client launch);
server.port / client.host appear only if set in the sweep config.

For a real run this script targets remote SSH boxes. Before every sweep it runs
script/vast_ports.sh --emit on the server box and sets server.port
(SPECEDGE_PORT) + client.host from the live VAST.ai port mapping. Pass
--no-discover-ports to skip that and read server.port / client.host from the
sweep config instead (e.g. a non-VAST box).

One-time setup flags, applied (in this order) before the sweep runs:
  --bootstrap-repo  clone paths.repo into remote_root on the server and client
                    boxes (fetch + ff-only if it already exists) and run
                    `uv sync` in each. Run this first -- the other setup steps
                    call scripts from the checkout.
  --bootstrap-ssh   keygen on the client box (script/ssh_key.sh), then append its
                    public key to ~/.ssh/authorized_keys on the server box and on
                    the client box itself.
Combine with --dry-run to do setup only.

Set `ssh_identity: ~/.ssh/vast` in the sweep config to have every remote ssh /
scp-equivalent call run as `ssh -i ~/.ssh/vast ...` (a per-box `ssh:` string
that already has its own -i wins).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
POLL = 5.0  # seconds between readiness / liveness polls
SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_REPO = "https://github.com/baguioni/specedge"

# Identity file injected as `ssh -i <path>` for every remote box, unless that
# box's own ssh string already carries a -i. Set from `ssh_identity` in main().
SSH_IDENTITY: str | None = None

_T0 = time.monotonic()


def _stamp() -> str:
    return f"{time.monotonic() - _T0:7.1f}s"


def info(msg: str, *a) -> None:
    print(f"[sweep {_stamp()}] {msg % a if a else msg}", flush=True)


def warn(msg: str, *a) -> None:
    print(f"[sweep {_stamp()}] WARNING: {msg % a if a else msg}", file=sys.stderr, flush=True)


def die(msg: str, *a) -> None:
    print(f"[sweep {_stamp()}] ERROR: {msg % a if a else msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, patch: dict | None) -> dict:
    """Recursively merge `patch` onto `base`, returning a new dict."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def render_config(sweep: dict, base_cfg: dict, experiment: dict) -> dict:
    cfg = deep_merge(base_cfg, sweep.get("common"))
    cfg = deep_merge(cfg, experiment.get("overrides"))

    cfg.setdefault("base", {})
    cfg.setdefault("server", {})
    cfg.setdefault("client", {})

    cfg["base"]["exp_name"] = experiment["name"]
    cfg["base"]["result_path"] = sweep["paths"]["result_path"]
    # server.port / client.host are populated by port discovery (the default) or,
    # with --no-discover-ports, taken straight from the sweep config; either way
    # they are set on `sweep` by the time we render. If neither supplied them,
    # keep whatever base_config already had.
    port = sweep.get("server", {}).get("port")
    host = sweep.get("client", {}).get("host")
    if port is not None:
        cfg["server"]["port"] = port
    if host:
        cfg["client"]["host"] = host
    return cfg


# --------------------------------------------------------------------------- #
# local / remote command execution
# --------------------------------------------------------------------------- #
def is_local(target: str) -> bool:
    return str(target).strip().lower() in ("", "local", "localhost")


def root_for(sweep: dict, target: str) -> str:
    return str(REPO) if is_local(target) else sweep["paths"]["remote_root"]


def ssh_argv(target: str, *, quiet: bool = False) -> list[str]:
    """`ssh` + the global identity (unless `target` already sets -i) + target args."""
    base = ["ssh"]
    if quiet:
        base.append("-q")
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
    """Run a bash snippet locally (`bash -lc`) or on an SSH target.

    When not capturing, stdio is inherited so long-running remote output (e.g.
    client_host.py) streams live to this terminal.
    """
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
# building blocks
# --------------------------------------------------------------------------- #
def put_config(target: str, remote_root: str, rel_path: str, content: str) -> None:
    # keep a local copy on this machine regardless of target
    local_copy = REPO / rel_path
    local_copy.parent.mkdir(parents=True, exist_ok=True)
    local_copy.write_text(content)

    if is_local(target):
        return

    remote_path = f"{remote_root.rstrip('/')}/{rel_path}"
    remote_dir = remote_path.rsplit("/", 1)[0]
    run(target, f"mkdir -p {remote_dir} && cat > {remote_path}",
        input_bytes=content.encode(), check=True, timeout=120)


def _targets(sweep: dict) -> list[str]:
    """server + client ssh targets, de-duplicated, preserving order."""
    out: list[str] = []
    for role in ("server", "client"):
        t = sweep[role]["ssh"]
        if t not in out:
            out.append(t)
    return out


def bootstrap_repo(sweep: dict) -> None:
    """Clone <paths.repo> into remote_root and run `uv sync` on each box.

    An existing checkout is fetched + fast-forwarded (never force-updated). A
    "local" target is left as-is apart from `uv sync`. `uv` and `git` are looked
    up via a login shell.
    """
    repo = str(sweep["paths"].get("repo") or DEFAULT_REPO)
    ref = str(sweep["paths"].get("repo_ref") or "").strip()

    for t in _targets(sweep):
        root = root_for(sweep, t)
        label = "local" if is_local(t) else t

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
                f"else mkdir -p \"$(dirname {root})\" && "
                f"git clone {shlex.quote(repo)} {root}{co_new}; fi"
            )
            info("%s: clone/update %s -> %s%s", label, repo, root, f" @ {ref}" if ref else "")
            run(t, f"bash -lc {shlex.quote(inner)}", check=True)

        info("%s: uv sync in %s", label, root)
        run(t, f"bash -lc {shlex.quote(f'cd {root} && uv sync')}", check=True)

    info("repo bootstrap complete")


def bootstrap_ssh(sweep: dict, base_cfg: dict) -> None:
    """Generate the client's SSH key and authorize it on the server box.

    Runs script/ssh_key.sh on the CLIENT box (idempotent keygen), reads the
    public half back, and appends it to ~/.ssh/authorized_keys on the SERVER box
    -- and on the client box itself, so `node: localhost`-style entries that
    client_host.py reaches over real SSH also work. Best-effort ssh-keyscan
    seeds the client's known_hosts so the first connection does not prompt.
    """
    client = sweep["client"]["ssh"]
    server = sweep["server"]["ssh"]
    croot = root_for(sweep, client)
    merged = deep_merge(base_cfg, sweep.get("common"))
    key_path = merged.get("base", {}).get("ssh_key", "~/.ssh/id_ed25519_server")

    if os.path.basename(key_path) != "id_ed25519_server":
        warn("base.ssh_key is %s but ssh_key.sh generates ~/.ssh/id_ed25519_server; "
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

    hosts: set[str] = set()
    server_host = str(sweep["client"].get("host", "")).rsplit(":", 1)[0].strip()
    if server_host:
        hosts.add(server_host)
    for node_name in (merged.get("node") or {}):
        h = str(node_name).split(":", 1)[0]
        if h not in ("localhost", "127.0.0.1", ""):
            hosts.add(h)
    for h in sorted(hosts):
        run(client, f"ssh-keyscan -T 5 {shlex.quote(h)} >> ~/.ssh/known_hosts 2>/dev/null || true")
    if hosts:
        run(client, "sort -u -o ~/.ssh/known_hosts ~/.ssh/known_hosts 2>/dev/null || true")
        info("client: known_hosts seeded for %s", ", ".join(sorted(hosts)))
    info("ssh bootstrap complete")


def discover_ports(sweep: dict) -> None:
    """Run script/vast_ports.sh --emit on the server box and overwrite
    server.port / client.host from the live VAST.ai port mapping.

    vast_ports.sh needs a login shell to see VAST_TCP_PORT_* / PUBLIC_IPADDR, so
    it is invoked via `bash -lc`.
    """
    target = sweep["server"]["ssh"]
    root = root_for(sweep, target)
    inner = f"cd {root} && bash script/vast_ports.sh --emit"
    r = run(target, f"bash -lc {shlex.quote(inner)}", capture=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"vast_ports.sh --emit failed on the server box (rc={r.returncode}): "
            f"{(r.stderr or b'').decode(errors='replace').strip()[:400]}\n"
            f"(pass --no-discover-ports to read server.port / client.host from the config)"
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
    if not (0 < int(kv["CONTAINER_PORT"]) < 65536 and 0 < int(kv["HOST_PORT"]) < 65536):
        raise RuntimeError(f"vast_ports.sh --emit returned out-of-range ports: {kv}")

    sweep["server"]["port"] = int(kv["CONTAINER_PORT"])
    sweep["client"]["host"] = f"{kv['PUBLIC_IP']}:{kv['HOST_PORT']}"
    info("discovered ports: server binds %s, client dials %s",
         kv["CONTAINER_PORT"], sweep["client"]["host"])


def start_server(sweep: dict, cfg_rel: str, exp: str) -> int:
    target = sweep["server"]["ssh"]
    root = root_for(sweep, target)
    result_rel = sweep["paths"]["result_path"]
    port = sweep["server"]["port"]
    out = f"{result_rel}/{exp}/server.stdout"

    # setsid + </dev/null so the launcher ssh returns immediately instead of
    # staying attached to the backgrounded server (which wedges `run()` forever).
    script = (
        f"cd {root} && "
        f"source .venv/bin/activate && "
        f"mkdir -p {result_rel}/{exp} && "
        f": > {out} && "
        f"SPECEDGE_PORT={port} setsid nohup python -O src/script/batch_server.py "
        f"--config {cfg_rel} </dev/null >{out} 2>&1 & echo $!"
    )
    info("launching server on %s ...", target if not is_local(target) else "local")
    try:
        r = run(target, script, capture=True, check=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"server launch on {target!r} did not return within 180s; "
            f"check {root}/{out} on the box"
        )
    lines = [ln.strip() for ln in out_of(r).splitlines() if ln.strip()]
    if not lines or not lines[-1].isdigit():
        raise RuntimeError(f"could not read server PID (stdout={out_of(r)!r}, stderr={(r.stderr or b'').decode()!r})")
    pid = int(lines[-1])
    info("server started on %s (pid %d), config %s", target if not is_local(target) else "local", pid, cfg_rel)
    return pid


def tail_server_log(sweep: dict, exp: str) -> subprocess.Popen | None:
    """Start a background `tail -F` of the server box's server.stdout, streaming
    it to this terminal with a `[server]` prefix. Returns the Popen to stop
    later, or None if it could not be started.
    """
    target = sweep["server"]["ssh"]
    root = root_for(sweep, target)
    fpath = f"{root}/{sweep['paths']['result_path']}/{exp}/server.stdout"
    pipeline = (
        f"tail -n +1 -F {shlex.quote(fpath)} 2>/dev/null "
        f"| awk '{{ print \"[server] \" $0; fflush() }}'"
    )
    argv = ["bash", "-lc", pipeline] if is_local(target) else [*ssh_argv(target, quiet=True), pipeline]
    try:
        proc = subprocess.Popen(argv)  # noqa: S603  (stdio inherited -> streams live)
    except OSError as e:
        warn("could not start server-log tail for %s: %s", exp, e)
        return None
    info("streaming server log for %s (lines prefixed [server])", exp)
    return proc


def stop_tail(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def pid_alive(target: str, pid: int) -> bool:
    r = run(target, f"kill -0 {pid} 2>/dev/null && echo Y || echo N", capture=True)
    return out_of(r).strip().endswith("Y")


def wait_for_ready(sweep: dict, exp: str, pid: int) -> None:
    target = sweep["server"]["ssh"]
    root = root_for(sweep, target)
    result_rel = sweep["paths"]["result_path"]
    marker = sweep["ready_marker"]
    timeout = sweep["timeouts"]["server_ready"]
    files = f"{root}/{result_rel}/{exp}/server.stdout {root}/{result_rel}/{exp}/server.log"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(target, pid):
            tail = out_of(run(target, f"tail -n 60 {files} 2>/dev/null", capture=True))
            raise RuntimeError(f"server pid {pid} exited before it was ready:\n{tail}")
        hit = run(target, f"grep -qsF {shlex.quote(marker)} {files} && echo OK || true", capture=True)
        if "OK" in out_of(hit):
            info("server ready for %s", exp)
            return
        time.sleep(POLL)
    raise TimeoutError(f"server not ready for {exp} within {timeout}s (no {marker!r} in server.stdout/log)")


def run_client_host(sweep: dict, cfg_rel: str) -> int:
    target = sweep["client"]["ssh"]
    root = root_for(sweep, target)
    timeout = sweep["timeouts"]["run"]
    script = (
        f"cd {root} && source .venv/bin/activate && "
        f"python src/script/client_host.py --config {cfg_rel}"
    )
    info("running client_host on %s ...", target if not is_local(target) else "local")
    try:
        return run(target, script, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        warn("client_host exceeded %ss; killing stray client processes", timeout)
        run(target, "pkill -f 'src/script/client_host.py' || true")
        run(target, "pkill -f 'src/script/client.py' || true")
        return -1


def stop_server(sweep: dict, pid: int) -> str:
    """Wait for the self-shutdown, then escalate. Returns how it died."""
    target = sweep["server"]["ssh"]
    timeout = sweep["timeouts"]["server_shutdown"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(target, pid):
            return "self"
        time.sleep(POLL)

    warn("server pid %d still up %ss after clients finished; sending SIGINT", pid, timeout)
    run(target, f"kill -INT {pid} 2>/dev/null || true")
    for _ in range(30):
        if not pid_alive(target, pid):
            return "sigint"
        time.sleep(1)

    warn("server pid %d ignored SIGINT; sending SIGKILL", pid)
    run(target, f"kill -KILL {pid} 2>/dev/null || true")
    time.sleep(2)
    return "sigkill" if not pid_alive(target, pid) else "LEAKED"


def collect(sweep: dict, exp: str) -> bool:
    collect_to = REPO / sweep["paths"]["collect_to"]
    collect_to.mkdir(parents=True, exist_ok=True)
    result_rel = sweep["paths"]["result_path"]

    ok = False
    for t in _targets(sweep):
        root = root_for(sweep, t)
        if is_local(t):
            src = Path(os.path.expanduser(f"{root}/{result_rel}/{exp}"))
            dst = collect_to / exp
            if not src.exists():
                continue
            if src.resolve() == dst.resolve():
                ok = True
                continue
            run("local", f"mkdir -p {shlex.quote(str(dst))} && "
                          f"cp -a {shlex.quote(str(src))}/. {shlex.quote(str(dst))}/", check=True)
            ok = True
        else:
            # tar over ssh: extract <exp>/ directly under collect_to/
            ssh_c = " ".join(shlex.quote(a) for a in ssh_argv(t))
            cmd = (
                f"{ssh_c} 'tar -C {root}/{result_rel} -czf - {exp}' "
                f"| tar -C {shlex.quote(str(collect_to))} -xzf -"
            )
            r = subprocess.run(["bash", "-lc", cmd])  # noqa: S603
            ok = ok or r.returncode == 0
            if r.returncode != 0:
                warn("could not pull %s results from %s", exp, t)
    return ok


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def select(experiments: list[dict], only: str | None, start_from: str | None) -> list[dict]:
    names = [e["name"] for e in experiments]
    if only:
        want = [s.strip() for s in only.split(",") if s.strip()]
        missing = [w for w in want if w not in names]
        if missing:
            die("unknown experiment(s): %s", ", ".join(missing))
        return [e for e in experiments if e["name"] in want]
    if start_from:
        if start_from not in names:
            die("unknown --from experiment: %s", start_from)
        return experiments[names.index(start_from):]
    return experiments


def run_one(sweep: dict, base_cfg: dict, experiment: dict, *, no_collect: bool,
            tail_server: bool = False) -> dict:
    exp = experiment["name"]
    rel = f"{sweep['paths']['rendered_dir'].rstrip('/')}/{exp}.yaml"
    rec: dict = {"name": exp, "status": "pending", "config": rel}

    cfg = render_config(sweep, base_cfg, experiment)
    content = yaml.safe_dump(cfg, sort_keys=False)

    t_start = time.time()
    pid = None
    tail = None
    try:
        put_config(sweep["server"]["ssh"], sweep["paths"]["remote_root"], rel, content)
        if sweep["client"]["ssh"] != sweep["server"]["ssh"]:
            put_config(sweep["client"]["ssh"], sweep["paths"]["remote_root"], rel, content)

        pid = start_server(sweep, rel, exp)
        if tail_server:
            tail = tail_server_log(sweep, exp)
        wait_for_ready(sweep, exp, pid)

        rc = run_client_host(sweep, rel)
        rec["client_rc"] = rc
        rec["status"] = "ok" if rc == 0 else "client_failed"
    except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as e:
        rec["status"] = "error"
        rec["error"] = str(e).splitlines()[0][:300]
        warn("%s: %s", exp, e)
    finally:
        if pid is not None:
            rec["shutdown"] = stop_server(sweep, pid)
        stop_tail(tail)

    rec["elapsed_s"] = round(time.time() - t_start, 1)

    if not no_collect:
        rec["collected"] = collect(sweep, exp)

    info("%s -> %s (%.0fs)", exp, rec["status"], rec["elapsed_s"])
    time.sleep(sweep["timeouts"]["cooldown"])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--config", default="config/sweep.example.yaml", help="sweep config (default: config/sweep.example.yaml)")
    ap.add_argument("--only", help="comma-separated experiment names to run")
    ap.add_argument("--from", dest="start_from", help="start at this experiment, run the rest")
    ap.add_argument("--dry-run", action="store_true",
                    help="render the merged per-experiment configs to rendered_dir and "
                         "exit; touches no box (no discovery, no launch)")
    ap.add_argument("--list", action="store_true", help="list experiment names and exit")
    ap.add_argument("--no-collect", action="store_true", help="skip pulling results back")
    ap.add_argument("--tail-server", action="store_true",
                    help="stream the server box's stdout to this terminal (lines "
                         "prefixed [server]) while each experiment runs")
    ap.add_argument("--bootstrap-repo", action="store_true",
                    help="clone <paths.repo> into remote_root and run `uv sync` on "
                         "the server and client boxes")
    ap.add_argument("--bootstrap-ssh", action="store_true",
                    help="run script/ssh_key.sh on the client box and authorize its "
                         "public key on the server box (and the client box itself)")
    ap.add_argument("--no-discover-ports", action="store_true",
                    help="skip automatic port discovery; read server.port / "
                         "client.host from the sweep config instead")
    args = ap.parse_args()

    sweep_path = Path(args.config)
    if not sweep_path.is_absolute():
        sweep_path = REPO / sweep_path
    if not sweep_path.is_file():
        die("no such sweep config: %s", sweep_path)
    sweep = yaml.safe_load(sweep_path.read_text())

    global SSH_IDENTITY
    if sweep.get("ssh_identity"):
        SSH_IDENTITY = os.path.expanduser(str(sweep["ssh_identity"]))
        if not Path(SSH_IDENTITY).is_file():
            warn("ssh_identity %s not found on this machine", SSH_IDENTITY)

    experiments = sweep.get("experiments") or []
    bad = [e["name"] for e in experiments if not SLUG_RE.match(e.get("name", ""))]
    if bad:
        die("experiment names must match %s: %s", SLUG_RE.pattern, ", ".join(bad))
    if len({e["name"] for e in experiments}) != len(experiments):
        die("duplicate experiment names")

    if args.list:
        for e in experiments:
            print(e["name"])
        return

    base_cfg_path = REPO / sweep["paths"]["base_config"]
    if not base_cfg_path.is_file():
        die("no such base_config: %s", base_cfg_path)
    base_cfg = yaml.safe_load(base_cfg_path.read_text())

    if args.bootstrap_repo:
        try:
            bootstrap_repo(sweep)
        except RuntimeError as e:
            die("%s", e)

    if args.dry_run:
        pass  # pure local render below -- no discovery, no remote calls
    elif args.no_discover_ports:
        missing = [
            f"{role}.{key}"
            for role, key in (("server", "port"), ("client", "host"))
            if not sweep.get(role, {}).get(key)
        ]
        if missing:
            die("--no-discover-ports needs %s set in the sweep config", ", ".join(missing))
    else:
        try:
            discover_ports(sweep)
        except RuntimeError as e:
            die("%s", e)

    if args.bootstrap_ssh:
        try:
            bootstrap_ssh(sweep, base_cfg)
        except RuntimeError as e:
            die("%s", e)

    chosen = select(experiments, args.only, args.start_from)
    info("%d experiment(s): %s", len(chosen), ", ".join(e["name"] for e in chosen))

    if args.dry_run:
        for e in chosen:
            cfg = render_config(sweep, base_cfg, e)
            rel = f"{sweep['paths']['rendered_dir'].rstrip('/')}/{e['name']}.yaml"
            out = REPO / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(cfg, sort_keys=False))
            info("rendered %s", rel)
        return

    for role in ("server", "client"):
        info("%-6s -> %s", role, sweep[role]["ssh"] if not is_local(sweep[role]["ssh"]) else "local")

    results: list[dict] = []
    try:
        for e in chosen:
            results.append(run_one(sweep, base_cfg, e, no_collect=args.no_collect,
                                   tail_server=args.tail_server))
    except KeyboardInterrupt:
        warn("interrupted; the current server (if any) was torn down in finally")

    summary = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sweep_config": str(sweep_path),
        "results": results,
    }
    dest = REPO / sweep["paths"]["collect_to"]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n==== sweep summary ====")
    for r in results:
        extra = r.get("error") or f"shutdown={r.get('shutdown', '-')}, collected={r.get('collected', '-')}"
        print(f"  {r['name']:<24} {r['status']:<14} {r['elapsed_s']:>7.0f}s  {extra}")
    print(f"  summary written to {dest / 'sweep_summary.json'}")

    if any(r["status"] not in ("ok",) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
