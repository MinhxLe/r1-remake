set dotenv-load

ssh:
    INSTANCE_ID=$(vastai show instances --raw | jq -r '.[0].id') \
    && echo $INSTANCE_ID \
    && SSH_PORT=$(vastai show instance $INSTANCE_ID --raw | jq -r '.direct_port_start') \
    && SSH_IP=$(vastai show instance $INSTANCE_ID --raw | jq -r '.public_ipaddr') \
    && ssh -p $SSH_PORT root@$SSH_IP
start_instance:
  ./scripts/setup_remote_instance.sh

end_instance:
  INSTANCE_ID=$(vastai show instances --raw | jq -r '.[0].id') \
  && vastai stop instance $INSTANCE_ID

sync:
  ./scripts/sync_remote_instance.sh
