#!/usr/bin/env bash
# vast_ports.sh — RUN THIS INSIDE THE VAST.AI INSTANCE (the target/server box).
#
# Automatically selects the first free TCP port published by vast.ai and exports
# SPECEDGE_PORT for batch_server.py. Prints port mapping info and setup instructions.
#
# Usage:
#   source ./vast_ports.sh  # export SPECEDGE_PORT
#   python src/script/batch_server.py --config config/config.yaml

set -u

pub="${PUBLIC_IPADDR:-}"
[ -z "$pub" ] && pub="$(curl -fsS --max-time 3 ifconfig.me 2>/dev/null || echo '<unknown>')"
echo "public IP : $pub"
echo

# vast.ai exposes each published port as VAST_TCP_PORT_<container>=<host>
mapfile -t rows < <(env | grep -oE '^VAST_TCP_PORT_[0-9]+=[0-9]+' | sort -t_ -k4 -n)

if [ "${#rows[@]}" -eq 0 ]; then
    echo "No VAST_TCP_PORT_* variables in this shell."
    echo "Run this from your LOCAL machine instead:"
    echo "  vastai show instance <ID> --raw \\"
    echo "    | jq -r '.ports | to_entries[] | \"\\(.key) -> \\(.value[0].HostPort)\"'"
    echo "  vastai show instance <ID> --raw | jq -r '.public_ipaddr'"
    exit 1
fi

snap="$(ss -tlnpH 2>/dev/null)"
busy_port() { printf '%s\n' "$snap" | awk '{print $4}' | sed -E 's/.*:([0-9]+)$/\1/' | grep -qx "$1"; }
who_port()  { printf '%s\n' "$snap" | awk -v p=":$1\$" '$4 ~ p' | grep -oE '"[^"]+"' | head -1 | tr -d '"'; }

printf '%-15s %-11s %-6s %s\n' "CONTAINER PORT" "HOST PORT" "STATE" "LISTENER"
printf '%-15s %-11s %-6s %s\n' "--------------" "---------" "-----" "--------"

selected_cport=""
selected_hport=""

for kv in "${rows[@]}"; do
    cport="${kv#VAST_TCP_PORT_}"; cport="${cport%%=*}"
    hport="${kv#*=}"
    if busy_port "$cport"; then
        printf '%-15s %-11s %-6s %s\n' "$cport" "$hport" "BUSY" "$(who_port "$cport")"
    else
        printf '%-15s %-11s %-6s %s\n' "$cport" "$hport" "FREE" "-"
        [ -z "$selected_cport" ] && selected_cport="$cport" && selected_hport="$hport"
    fi
done

echo
if [ -z "$selected_cport" ]; then
    echo "ERROR: No free ports available."
    exit 1
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
