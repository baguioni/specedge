"""Tell a persistent batch server to shut down.

A sweep now starts one ``batch_server.py`` and reuses it for every experiment:
clients send ``Sync`` at the start of each run, which re-points the server's
result logger at that run's folder. Nothing trips the server's shutdown on its
own any more, so the orchestrator calls this once it has run the last
experiment (SIGINT on the server process works too).

Usage:
  python src/script/stop_batch_server.py --host 127.0.0.1:8080
"""

import argparse

import grpc

from specedge_grpc import specedge_pb2, specedge_pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", required=True, help="server address as host:port"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="RPC timeout in seconds"
    )
    args = parser.parse_args()

    with grpc.insecure_channel(args.host) as channel:
        stub = specedge_pb2_grpc.SpecEdgeServiceStub(channel)
        stub.Done(
            specedge_pb2.DoneRequest(client_idx=-1, shutdown=True),
            timeout=args.timeout,
        )

    print(f"shutdown requested: {args.host}")


if __name__ == "__main__":
    main()
