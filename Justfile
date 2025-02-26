set dotenv-load

ssh:
  ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_IP

sync_env:
  scp -P $REMOTE_PORT .env ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PROJECT_DIR} 

sync:
    rsync -avz \
    --exclude-from=.gitignore \
    --exclude=.git \
    --include=".env" \
    -e "ssh -p ${REMOTE_PORT}" \
    . ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PROJECT_DIR}

watch_sync:
    fswatch -o \
    --exclude "\.git" \
    --exclude "\.idea" \
    --exclude "node_modules" \
    . | xargs -n1 -I{} just sync
