#!/usr/bin/env python
import sys
import yaml
import argparse
from src.es_drl.dpg.td3_trainer import TD3Trainer


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune TD3, optionally from ES checkpoint"
    )
    parser.add_argument("common_config", help="Path to common.yaml")
    parser.add_argument("td3_config", help="Path to td3_finetune.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrained", help="Path to ES checkpoint file (.pt)")
    group.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load any ES checkpoint (train TD3 from scratch)",
    )

    args = parser.parse_args()

    common_cfg = load_yaml(args.common_config)
    td3_cfg = load_yaml(args.td3_config)

    if args.no_pretrained:
        pretrained_path = None
    else:
        pretrained_path = args.pretrained

    trainer = TD3Trainer(common_cfg, td3_cfg, pretrained_path=pretrained_path)

    trainer.train()
