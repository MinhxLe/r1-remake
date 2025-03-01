#!/bin/bash
REMOTE_PROJECT_DIR="~/r1"
# create directory if does not exist
INSTANCE_ID=${1:-$(vastai show instances --raw | jq -r '.[0].id')}
echo "Syncing instance $INSTANCE_ID"
INSTANCE_IP=$(vastai show instance $INSTANCE_ID --raw | jq -r '.public_ipaddr')
INSTANCE_PORT=$(vastai show instance $INSTANCE_ID --raw | jq -r '.direct_port_start')

ssh -p $INSTANCE_PORT root@$INSTANCE_IP  'mkdir -p $REMOTE_PROJECT_DIR'

# syncing code
rsync -avz \
--exclude-from=.gitignore \
--exclude=.git \
--include=".env" \
-e "ssh -p $INSTANCE_PORT" . root@$INSTANCE_IP:$REMOTE_PROJECT_DIR

# syncing .env
scp -P $INSTANCE_PORT .env root@$INSTANCE_IP:$REMOTE_PROJECT_DIR

# syncing uv [TODO] make this work
# ssh -p $INSTANCE_PORT root@$INSTANCE_IP 'source $HOME/.local/bin/env && cd ~/$REMOTE_PROJECT_DIR && uv sync'
