#!/usr/bin/env bash
# vast_ports.sh — RUN THIS INSIDE THE VAST.AI INSTANCE (the target/server box).
#
# Automatically selects the first free TCP port published by vast.ai and exports
# SPECEDGE_PORT for batch_server.py. Prints port mapping info and setup instructions.
#
# Usage:
#   source ./vast_ports.sh          # export SPECEDGE_PORT, print human instructions
#   python src/script/batch_server.py --config config/config.yaml
#
#   ./vast_ports.sh --emit          # machine-readable: only KEY=VALUE lines on stdout
#       PUBLIC_IP=<ip>
#       CONTAINER_PORT=<port batch_server binds / SPECEDGE_PORT>
#       HOST_PORT=<port the client dials, i.e. client.host = PUBLIC_IP:HOST_PORT>
#   Consumed by script/sweep.py --discover-ports. All diagnostics go to stderr;
#   a non-zero exit means discovery failed.

set -u

emit=0
for arg in "$@"; do
    case "$arg" in
        --emit) emit=1 ;;
        -h|--help)
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

# In --emit mode every human-facing line goes to stderr so stdout stays clean.
say() {
    if [ "$emit" = 1 ]; then
        printf '%s\n' "$*" >&2
    else
        printf '%s\n' "$*"
    fi
}

pub="${PUBLIC_IPADDR:-}"
[ -z "$pub" ] && pub="$(curl -fsS --max-time 3 ifconfig.me 2>/dev/null || echo '<unknown>')"
say "public IP : $pub"
say ""

# vast.ai exposes each published port as VAST_TCP_PORT_<container>=<host>
rows=()
while IFS= read -r line; do
    rows+=("$line")
done < <(env | grep -oE '^VAST_TCP_PORT_[0-9]+=[0-9]+' | sort -t_ -k4 -n)

if [ "${#rows[@]}" -eq 0 ]; then
    echo "No VAST_TCP_PORT_* variables in this shell." >&2
    echo "Run this inside the instance via a login shell (bash -lc), or from your" >&2
    echo "LOCAL machine resolve them with:" >&2
    echo "  vastai show instance <ID> --raw \\" >&2
    echo "    | jq -r '.ports | to_entries[] | \"\\(.key) -> \\(.value[0].HostPort)\"'" >&2
    echo "  vastai show instance <ID> --raw | jq -r '.public_ipaddr'" >&2
    exit 1
fi

snap="$(ss -tlnpH 2>/dev/null)"
busy_port() { printf '%s\n' "$snap" | awk '{print $4}' | sed -E 's/.*:([0-9]+)$/\1/' | grep -qx "$1"; }
who_port()  { printf '%s\n' "$snap" | awk -v p=":$1\$" '$4 ~ p' | grep -oE '"[^"]+"' | head -1 | tr -d '"'; }

if [ "$emit" != 1 ]; then
    printf '%-15s %-11s %-6s %s\n' "CONTAINER PORT" "HOST PORT" "STATE" "LISTENER"
    printf '%-15s %-11s %-6s %s\n' "--------------" "---------" "-----" "--------"
fi

selected_cport=""
selected_hport=""

for kv in "${rows[@]}"; do
    cport="${kv#VAST_TCP_PORT_}"; cport="${cport%%=*}"
    hport="${kv#*=}"
    if busy_port "$cport"; then
        [ "$emit" = 1 ] || printf '%-15s %-11s %-6s %s\n' "$cport" "$hport" "BUSY" "$(who_port "$cport")"
    else
        [ "$emit" = 1 ] || printf '%-15s %-11s %-6s %s\n' "$cport" "$hport" "FREE" "-"
        [ -z "$selected_cport" ] && selected_cport="$cport" && selected_hport="$hport"
    fi
done

say ""
if [ -z "$selected_cport" ]; then
    echo "ERROR: No free published ports available." >&2
    exit 1
fi

if [ "$emit" = 1 ]; then
    printf 'PUBLIC_IP=%s\n' "$pub"
    printf 'CONTAINER_PORT=%s\n' "$selected_cport"
    printf 'HOST_PORT=%s\n' "$selected_hport"
    exit 0
fi

export SPECEDGE_PORT="$selected_cport"
echo "✓ Auto-selected first FREE port: $selected_cport (host: $selected_hport)"
echo "  SPECEDGE_PORT=$selected_cport (exported)"
echo
echo "Next steps:"
echo "  1. Update client config.yaml:"
echo "     client.host: $pub:$selected_hport"
echo
echo "  2. Run batch_server.py (SPECEDGE_PORT is already exported):"
echo "     python src/script/batch_server.py --config config/config.yaml"
echo
echo "Reachability test to run FROM THE CLIENT:"
echo "  nc -vz $pub $selected_hport"
