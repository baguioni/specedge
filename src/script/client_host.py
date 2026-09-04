import argparse
import subprocess
import sys
from pathlib import Path

import yaml

import log

SPECEDGE_ROOT = Path(__file__).absolute().parents[2]


def main(config_file: str):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # base configuration
    result_path = config["base"]["result_path"]
    exp_name = config["base"]["exp_name"]
    dtype = config["base"]["dtype"]
    seed = config["base"]["seed"]
    ssh_key = str(Path(config["base"]["ssh_key"]).expanduser())
    optimization = config["opt"]
    max_len = config["base"]["max_len"]

    log_config = log.get_default_log_config(Path(result_path) / exp_name, "client_host")
    log.configure_logging(log_config)
    log.log_unexpected_exception()

    logger = log.get_logger()

    logger.info("Starting client host")

    logger.debug("result_path: %s", result_path)
    logger.debug("exp_name: %s", exp_name)

    # client configuration
    host = config["client"]["host"]
    base_process_name = config["client"]["process_name"]
    draft_model = config["client"]["draft_model"]
    dataset = config["client"]["dataset"]
    max_n_beams = config["client"]["max_n_beams"]
    max_beam_len = config["client"]["max_beam_len"]
    max_branch_width = config["client"]["max_branch_width"]
    max_budget = config["client"]["max_budget"]
    req_offset = config["client"]["req_offset"]
    sample_req_cnt = config["client"]["sample_req_cnt"]
    reasoning = config["client"]["reasoning"]

    logger.debug("draft_model: %s", draft_model)
    logger.debug("dataset: %s", dataset)
    logger.debug("max_n_beams: %s", max_n_beams)
    logger.debug("max_beam_len: %s", max_beam_len)
    logger.debug("max_branch_width: %s", max_branch_width)
    logger.debug("max_budget: %s", max_budget)
    logger.debug("reasoning: %s", reasoning)

    # client praoctive draft configuration
    proactive_cfg = config["client"]["proactive"]
    proactive_type = proactive_cfg["type"]
    proactive_max_n_beams = proactive_cfg["max_n_beams"]
    proactive_max_beam_len = proactive_cfg["max_beam_len"]
    proactive_max_branch_width = proactive_cfg["max_branch_width"]
    proactive_max_budget = proactive_cfg["max_budget"]

    logger.debug("proactive_type: %s", proactive_type)
    logger.debug("proactive_max_n_beams: %s", proactive_max_n_beams)
    logger.debug("proactive_max_beam_len: %s", proactive_max_beam_len)
    logger.debug("proactive_max_branch_width: %s", proactive_max_branch_width)
    logger.debug("proactive_max_budget: %s", proactive_max_budget)

    # overlap strategy configuration (proactive draft vs. saguaro cache)
    overlap_strategy = proactive_cfg.get(
        "strategy", "disabled" if proactive_type == "disabled" else "proactive"
    )
    saguaro_cfg = proactive_cfg.get("saguaro", {}) or {}
    saguaro_budget = saguaro_cfg.get("budget", 12)
    saguaro_branch_len = saguaro_cfg.get("branch_len", proactive_max_beam_len)
    saguaro_fan_out = saguaro_cfg.get("fan_out", "geometric")
    saguaro_init_accept_rate = saguaro_cfg.get("init_accept_rate", 0.5)
    saguaro_linear = saguaro_cfg.get("linear", "auto")

    logger.debug("overlap_strategy: %s", overlap_strategy)
    logger.debug("saguaro_budget: %s", saguaro_budget)
    logger.debug("saguaro_branch_len: %s", saguaro_branch_len)

    # experiment configuration
    max_new_tokens = config["client"]["max_new_tokens"]
    max_request_num = config["client"]["max_request_num"]

    # node configuration
    nodes = config["node"]
    logger.debug("nodes: %s", nodes)

    ssh_processes = {}
    client_idx = 0
    for node_name, node_info in nodes.items():
        for client_info in node_info:
            device = client_info["device"]

            logger.info("Starting a client_%s on %s, %s", client_idx, node_name, device)

            env_vars = {
                "SPECEDGE_OPTIMIZATION": optimization,
                "SPECEDGE_RESULT_PATH": result_path,
                "SPECEDGE_EXP_NAME": exp_name,
                "SPECEDGE_PROCESS_NAME": f"{base_process_name}_{client_idx}",
                "SPECEDGE_SEED": seed,
                "SPECEDGE_MAX_LEN": max_len,
                "SPECEDGE_DRAFT_MODEL": draft_model,
                "SPECEDGE_DEVICE": device,
                "SPECEDGE_DTYPE": dtype,
                "SPECEDGE_DATASET": dataset,
                "SPECEDGE_MAX_N_BEAMS": max_n_beams,
                "SPECEDGE_MAX_BEAM_LEN": max_beam_len,
                "SPECEDGE_MAX_BRANCH_WIDTH": max_branch_width,
                "SPECEDGE_MAX_BUDGET": max_budget,
                "SPECEDGE_PROACTIVE_TYPE": proactive_type,
                "SPECEDGE_PROACTIVE_MAX_N_BEAMS": proactive_max_n_beams,
                "SPECEDGE_PROACTIVE_MAX_BEAM_LEN": proactive_max_beam_len,
                "SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH": proactive_max_branch_width,
                "SPECEDGE_PROACTIVE_MAX_BUDGET": proactive_max_budget,
                "SPECEDGE_OVERLAP_STRATEGY": overlap_strategy,
                "SPECEDGE_SAGUARO_BUDGET": saguaro_budget,
                "SPECEDGE_SAGUARO_BRANCH_LEN": saguaro_branch_len,
                "SPECEDGE_SAGUARO_FAN_OUT": saguaro_fan_out,
                "SPECEDGE_SAGUARO_INIT_ACCEPT_RATE": saguaro_init_accept_rate,
                "SPECEDGE_SAGUARO_LINEAR": saguaro_linear,
                "SPECEDGE_MAX_NEW_TOKENS": max_new_tokens,
                "SPECEDGE_MAX_REQUEST_NUM": max_request_num,
                "SPECEDGE_REQ_OFFSET": req_offset,
                "SPECEDGE_SAMPLE_REQ_CNT": sample_req_cnt,
                "SPECEDGE_HOST": host,
                "SPECEDGE_CLIENT_IDX": client_idx,
                "SPECEDGE_REASONING": reasoning,
            }

            cmd = f"cd {SPECEDGE_ROOT} && "

            for key, value in env_vars.items():
                cmd += f'export {key}="{value}" && '

            cmd += "bash ./script/client.sh"

            logger.debug("cmd: %s", cmd)

            # node_name may be "host" or "host:port" (e.g. VAST.ai remaps SSH
            # off the default port 22, so a custom port must be specified)
            ssh_host, _, ssh_port = node_name.partition(":")

            if ssh_host in ["localhost", "127.0.0.1"]:
                logger.debug("Running locally on %s (no SSH)", node_name)
                process = subprocess.Popen(  # noqa: S603
                    ["bash", "-c", cmd],  # noqa: S607
                    stdout=subprocess.PIPE,
                    stderr=sys.stderr.buffer,
                    text=True,
                )
            else:
                ssh_cmd = ["ssh", "-i", ssh_key]
                if ssh_port:
                    ssh_cmd += ["-p", ssh_port]
                ssh_cmd += [ssh_host, cmd]

                logger.debug("Running via SSH on %s (port %s)", ssh_host, ssh_port or 22)
                process = subprocess.Popen(  # noqa: S603
                    ssh_cmd,  # noqa: S607
                    stdout=subprocess.PIPE,
                    stderr=sys.stderr.buffer,
                    text=True,
                )

            ssh_processes[client_idx] = process
            client_idx += 1

    for client_idx, process in ssh_processes.items():
        process.wait()
        logger.info("client_%d finished", client_idx)

    logger.info("All clients finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    main(args.config)
