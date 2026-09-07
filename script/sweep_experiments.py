#!/usr/bin/env python3
"""Client-side experiment sweep for SpecEdge against a persistent server.

Runs on the CLIENT box. The batch server is assumed to be already up and to
stay up for the whole sweep -- clients Sync at the start of each experiment
(the server re-points its result logger to result/<exp_name>/ and drops stale
KV state) and Done at the end (the server re-arms for the next run). This
script never starts or stops the server.

For each experiment in the sweep config it:

  1. renders  config = base_config <- common <- experiment.overrides, with
     base.exp_name set to the experiment name, base.result_path forced from the
     sweep config, and client.host forced to the server address to dial;
  2. writes the rendered config under paths.rendered_dir;
  3. runs  src/script/client_host.py --config <rendered>  and waits for it to
     exit (client_host.py spawns the per-node clients itself);
  4. gathers result/<exp_name>/ into paths.collect_to/<exp_name>/ -- and, when
     server.ssh is set, also pulls the server-side result/<exp_name>/server.*
     so `python src/metric/specedge.py -d <folder>` works straight away.

Shut the server down yourself once the sweep is done:
  python src/script/stop_batch_server.py --host <host:port>

Assumptions (must hold for every experiment, since the server is not restarted):
  the running server's target_model, device, dtype, max_len, max_batch_size,
  num_clients, batch_type, temperature, seed, dataset/req_offset/sample_req_cnt
  and the client max_n_beams / max_budget it was booted with are what every
  rendered config also uses. Only client-side knobs (proactive/saguaro params,
  draft_model, max_new_tokens, max_request_num) may vary between experiments.

Usage:
  python script/sweep_experiments.py -f config/sweep.example.yaml
  python script/sweep_experiments.py -f config/sweep.example.yaml --host 1.2.3.4:8080
  python script/sweep_experiments.py -f config/sweep.example.yaml --list
  python script/sweep_experiments.py -f config/sweep.example.yaml --only proactive,saguaro_geom
  python script/sweep_experiments.py -f config/sweep.example.yaml --from saguaro_geom
  python script/sweep_experiments.py -f config/sweep.example.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_RENDERED_DIR = "config/_sweep"

_T0 = time.monotonic()


def _stamp() -> str:
    return f"{time.monotonic() - _T0:7.1f}s"


def info(msg: str, *a) -> None:
    print(f"[sweep {_stamp()}] {msg % a if a else msg}", flush=True)


def warn(msg: str, *a) -> None:
    print(
        f"[sweep {_stamp()}] WARNING: {msg % a if a else msg}",
        file=sys.stderr,
        flush=True,
    )


def die(msg: str, *a) -> None:
    print(
        f"[sweep {_stamp()}] ERROR: {msg % a if a else msg}",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# config rendering
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


def render_config(sweep: dict, base_cfg: dict, experiment: dict, host: str) -> dict:
    cfg = deep_merge(base_cfg, sweep.get("common"))
    cfg = deep_merge(cfg, experiment.get("overrides"))

    cfg.setdefault("base", {})
    cfg.setdefault("client", {})

    cfg["base"]["exp_name"] = experiment["name"]
    cfg["base"]["result_path"] = sweep["paths"]["result_path"]
    cfg["client"]["host"] = host
    return cfg


def write_rendered(sweep: dict, base_cfg: dict, experiment: dict, host: str) -> str:
    rendered_dir = str(sweep["paths"].get("rendered_dir", DEFAULT_RENDERED_DIR))
    rel = f"{rendered_dir.rstrip('/')}/{experiment['name']}.yaml"
    out = REPO / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(render_config(sweep, base_cfg, experiment, host), sort_keys=False)
    )
    return rel


# --------------------------------------------------------------------------- #
# experiment selection
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


# --------------------------------------------------------------------------- #
# running one experiment
# --------------------------------------------------------------------------- #
def run_client_host(rendered_rel: str, timeout: float) -> int:
    """Run src/script/client_host.py for one experiment; return its exit code."""
    argv = [sys.executable, "src/script/client_host.py", "--config", rendered_rel]
    info("client_host --config %s", rendered_rel)
    try:
        return subprocess.run(argv, cwd=REPO, timeout=timeout).returncode  # noqa: S603
    except subprocess.TimeoutExpired:
        warn("client_host exceeded %ss; killing stray client processes", timeout)
        for pat in ("src/script/client_host.py", "src/script/client.py"):
            subprocess.run(["pkill", "-f", pat])  # noqa: S603, S607
        return -1


def _server_ssh_argv(sweep: dict) -> list[str] | None:
    """`ssh [-i identity] <server ssh args>`, or None if server.ssh is unset."""
    server_ssh = (sweep.get("server") or {}).get("ssh")
    if not server_ssh:
        return None
    argv = ["ssh"]
    ident = sweep.get("ssh_identity")
    if ident and " -i " not in f" {server_ssh} ":
        argv += ["-i", str(Path(ident).expanduser())]
    return [*argv, *shlex.split(str(server_ssh))]


def collect(sweep: dict, exp: str) -> bool:
    """Gather result/<exp>/ into collect_to/<exp>/.

    Always copies the local (client-side) results. When server.ssh is set, also
    tars the server-side result/<exp>/ back so server.jsonl lands next to the
    client_*.jsonl and the metric script can read the folder directly. The
    server-side pull assumes the running server writes under the same
    paths.result_path (relative to paths.remote_root).
    """
    result_path = str(sweep["paths"]["result_path"])
    collect_to = (REPO / sweep["paths"].get("collect_to", result_path)).resolve()
    src = REPO / result_path / exp
    dst = collect_to / exp

    ok = False
    if src.exists():
        if src.resolve() != dst.resolve():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        ok = True
    else:
        warn("%s: no local client results at %s", exp, src)

    ssh = _server_ssh_argv(sweep)
    remote_root = (sweep.get("paths") or {}).get("remote_root")
    if ssh and remote_root:
        rroot = f"{str(remote_root).rstrip('/')}/{result_path}"
        collect_to.mkdir(parents=True, exist_ok=True)
        ssh_str = " ".join(shlex.quote(a) for a in ssh)
        cmd = (
            f"{ssh_str} 'tar -C {shlex.quote(rroot)} -czf - {shlex.quote(exp)}' "
            f"| tar -C {shlex.quote(str(collect_to))} -xzf -"
        )
        r = subprocess.run(["bash", "-c", cmd])  # noqa: S603, S607
        if r.returncode == 0:
            info("%s: pulled server results from %s", exp, sweep["server"]["ssh"])
            ok = True
        else:
            warn("%s: could not pull server results (rc=%d)", exp, r.returncode)
    elif ssh and not remote_root:
        warn("server.ssh is set but paths.remote_root is missing; skipping server pull")

    return ok


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-f", "--config", default="config/sweep.example.yaml", help="sweep config"
    )
    ap.add_argument(
        "--host", help="server address host:port (overrides client.host in the config)"
    )
    ap.add_argument("--only", help="comma-separated experiment names to run")
    ap.add_argument("--from", dest="start_from", help="start at this experiment, run the rest")
    ap.add_argument("--list", action="store_true", help="list experiment names and exit")
    ap.add_argument("--dry-run", action="store_true", help="render configs and exit")
    ap.add_argument("--no-collect", action="store_true", help="do not gather results")
    args = ap.parse_args()

    sweep_path = Path(args.config)
    if not sweep_path.is_absolute():
        sweep_path = REPO / sweep_path
    if not sweep_path.is_file():
        die("no such sweep config: %s", sweep_path)
    sweep = yaml.safe_load(sweep_path.read_text())

    experiments = sweep.get("experiments") or []
    if not experiments:
        die("no experiments defined in %s", sweep_path)
    bad = [e.get("name", "") for e in experiments if not SLUG_RE.match(e.get("name", ""))]
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

    host = args.host or (sweep.get("client") or {}).get("host")
    if not host:
        die("no server address: set client.host in the sweep config or pass --host")

    run_timeout = float((sweep.get("timeouts") or {}).get("run", 5400))
    cooldown = float((sweep.get("timeouts") or {}).get("cooldown", 10))

    chosen = select(experiments, args.only, args.start_from)
    if not chosen:
        die("no experiments selected")
    info("%d experiment(s): %s", len(chosen), ", ".join(e["name"] for e in chosen))
    info("server (assumed already running): %s", host)

    if args.dry_run:
        for e in chosen:
            info("rendered %s", write_rendered(sweep, base_cfg, e, host))
        return

    results: list[dict] = []
    try:
        for i, e in enumerate(chosen):
            exp = e["name"]
            rel = write_rendered(sweep, base_cfg, e, host)

            rec: dict = {"name": exp, "status": "pending", "config": rel}
            t_start = time.time()
            rc = run_client_host(rel, run_timeout)
            rec["client_rc"] = rc
            rec["status"] = "ok" if rc == 0 else "client_failed"
            rec["elapsed_s"] = round(time.time() - t_start, 1)

            if not args.no_collect:
                rec["collected"] = collect(sweep, exp)

            info("%s -> %s (%.0fs)", exp, rec["status"], rec["elapsed_s"])
            results.append(rec)

            if i != len(chosen) - 1 and cooldown:
                time.sleep(cooldown)
    except KeyboardInterrupt:
        warn("interrupted")

    summary = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sweep_config": str(sweep_path),
        "host": host,
        "results": results,
    }
    dest = REPO / sweep["paths"].get("collect_to", sweep["paths"]["result_path"])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n==== sweep summary ====")
    for r in results:
        print(
            f"  {r['name']:<28} {r['status']:<14} "
            f"{r.get('elapsed_s', 0):>7.0f}s  collected={r.get('collected', '-')}"
        )
    print(f"  summary written to {dest / 'sweep_summary.json'}")

    if any(r["status"] != "ok" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
