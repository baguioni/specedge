#!/usr/bin/env bash
# ssh_key.sh — Generate SSH key, add to ssh-agent, and display public key

set -e

KEY_PATH="$HOME/.ssh/id_25519_server"
KEY_DIR="$(dirname "$KEY_PATH")"

# Create .ssh directory if it doesn't exist
if [ ! -d "$KEY_DIR" ]; then
    mkdir -p "$KEY_DIR"
    chmod 700 "$KEY_DIR"
    echo "✓ Created $KEY_DIR"
fi

# Generate key if it doesn't exist
if [ ! -f "$KEY_PATH" ]; then
    echo "Generating SSH key at $KEY_PATH..."
    ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "specedge-server"
    chmod 600 "$KEY_PATH"
    chmod 644 "$KEY_PATH.pub"
    echo "✓ SSH key generated"
else
    echo "✓ SSH key already exists at $KEY_PATH"
fi

# Add to ssh-agent
echo "Adding key to ssh-agent..."
eval "$(ssh-agent -s)" > /dev/null 2>&1 || true
ssh-add "$KEY_PATH" 2>/dev/null || {
    echo "Starting ssh-agent..."
    eval "$(ssh-agent -s)"
    ssh-add "$KEY_PATH"
}
echo "✓ Key added to ssh-agent"

echo
echo "========== PUBLIC KEY (copy-paste into server) =========="
cat "$KEY_PATH.pub"
echo "=========================================================="
echo
echo "Key location: $KEY_PATH"
