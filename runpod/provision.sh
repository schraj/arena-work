#!/usr/bin/env bash
# Run this LOCALLY (macOS). Creates the persistent volume + the GPU pod, waits for SSH,
# and prints the ~/.ssh/config block for VS Code / Cursor Remote-SSH.
set -euo pipefail

NAME="${NAME:-arena-dev}"
DC="${DC:-EU-RO-1}"                                   # volume is DC-locked; pod must match
GPU="${GPU:-NVIDIA GeForce RTX 4090}"                 # sm_89 — safe for pinned jax 0.10.1 / brax 0.14.2
TEMPLATE="${TEMPLATE:-runpod-torch-v280}"             # runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
VOL_GB="${VOL_GB:-100}"
STOP_AFTER="${STOP_AFTER:-$(date -u -v+8H +%FT%TZ)}"  # cost guard: auto-STOP (not delete) in 8h

command -v runpodctl >/dev/null || { echo "runpodctl not found"; exit 1; }
runpodctl ssh list-keys | grep -q '"key"' || {
  echo "No SSH key on the account. Keys are injected at BOOT, so register one first:"
  echo "  runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub"; exit 1; }

echo "==> Creating ${VOL_GB}GB network volume in ${DC}"
VOL_ID=$(runpodctl network-volume create --name "${NAME}-vol" --size "$VOL_GB" \
           --data-center-id "$DC" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
echo "    volume: $VOL_ID"

echo "==> Creating pod ($GPU, $DC), auto-stop at $STOP_AFTER"
POD_ID=$(runpodctl pod create --name "$NAME" \
  --template-id "$TEMPLATE" --gpu-id "$GPU" \
  --data-center-ids "$DC" --cloud-type SECURE \
  --network-volume-id "$VOL_ID" --volume-mount-path /workspace \
  --container-disk-in-gb 40 \
  --ports "22/tcp,8888/http" \
  --ssh --stop-after "$STOP_AFTER" \
  --wait --wait-timeout 10m | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
echo "    pod: $POD_ID"

# `--wait` only proves an SSH banner answers; ssh info gives the real IP/port/key.
INFO=$(runpodctl ssh info "$POD_ID")
IP=$(echo   "$INFO" | python3 -c 'import json,sys;print(json.load(sys.stdin)["ip"])')
PORT=$(echo "$INFO" | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
KEY=$(echo  "$INFO" | python3 -c 'import json,sys;print(json.load(sys.stdin)["ssh_key"]["path"])')

cat <<EOF

=== Pod ready: $POD_ID (volume $VOL_ID) ===

Add to ~/.ssh/config  (Port CHANGES after every stop/start — re-read with 'runpodctl ssh info $POD_ID'):

Host arena-pod
    HostName $IP
    User root
    Port $PORT
    IdentityFile $KEY
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Next:
  scp -P $PORT -i $KEY runpod/pod-install.sh runpod/pod-bootstrap.sh root@$IP:/workspace/
  ssh arena-pod 'bash /workspace/pod-install.sh'     # one-time, ~15 min
EOF
