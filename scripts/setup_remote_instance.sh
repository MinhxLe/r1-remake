#!/bin/bash

# start instance if not started
INSTANCE_ID="$(vastai show instances --raw | jq '.[0].id')"
vastai start instance $INSTANCE_ID

while [[ "$(vastai show instance $INSTANCE_ID --raw | jq -r '.actual_status')" != "running" ]]; do
    echo "Waiting for instance to be running..."
    sleep 10  # Wait for 10 seconds before checking again
done
echo "Instance is now running"

# id for minh ssh
vastai attach ssh $INSTANCE_ID "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGNUr5z1YufAaVBGoqemW5gEDsP9/FwXkHXio5DeUCps minh.d.le27@gmail.com"

INSTANCE_IP=$(vastai show instance $INSTANCE_ID --raw | jq -r '.public_ipaddr')
INSTANCE_PORT=$(vastai show instance $INSTANCE_ID --raw | jq -r '.direct_port_end')

# set up uv
ssh -p $INSTANCE_PORT root@$INSTANCE_IP "curl -LsSf https://astral.sh/uv/install.sh | sh"
ssh -p $INSTANCE_PORT root@$INSTANCE_IP "touch ~/.no_auto_tmux"


# add ssh keys
