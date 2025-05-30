#!/usr/bin/env bash
#
# Usage:
#   run_finetune.sh <common.yaml> <td3_finetune.yaml> [<es_checkpoint.pt> | --no-pretrained]
#

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: run_finetune.sh <common.yaml> <td3_finetune.yaml> [<es_checkpoint.pt> | --no-pretrained]"
  exit 1
fi

COMMON_CFG=$1
TD3_CFG=$2

if [ "$#" -eq 3 ]; then
  if [ "$3" = "--no-pretrained" ]; then
    python -m src.es_drl.main_finetune \
      "$COMMON_CFG" "$TD3_CFG" --no-pretrained
  else
    python -m src.es_drl.main_finetune \
      "$COMMON_CFG" "$TD3_CFG" --pretrained "$3"
  fi
else
  # Only two args: default to --no-pretrained
  python -m src.es_drl.main_finetune \
    "$COMMON_CFG" "$TD3_CFG" --no-pretrained
fi
