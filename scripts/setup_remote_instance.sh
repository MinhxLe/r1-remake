#!/bin/bash
ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_IP  'mkdir ~/$REMOTE_PROJECT_DIR'

just sync_env
just sync

ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_IP 'curl -LsSf https://astral.sh/uv/install.sh | sh'

ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_IP 'uv sync'
