#!/usr/bin/env bash
CONFIGS=("$@")
python -m src.es_drl.main_es "${CONFIGS[@]}"
