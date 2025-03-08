set dotenv-load

ssh:
    INSTANCE_ID=$(vastai show instances --raw | jq -r '.[0].id') \
    && echo $INSTANCE_ID \
    && SSH_PORT=$(vastai show instance $INSTANCE_ID --raw | jq -r '.ports."22/tcp"[0].HostPort') \
    && echo $SSH_PORT \
    && SSH_IP=$(vastai show instance $INSTANCE_ID --raw | jq -r '.public_ipaddr') \
    && echo $SSH_IP \
    && ssh -p $SSH_PORT root@$SSH_IP
start_instance:
  ./scripts/setup_remote_instance.sh

end_instance:
  INSTANCE_ID=$(vastai show instances --raw | jq -r '.[0].id') \
  && vastai stop instance $INSTANCE_ID

sync:
  ./scripts/sync_remote_instance.sh
