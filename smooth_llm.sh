#!/bin/bash

perturbation_type='RandomSwapPerturbation'
perturbation_percentage=10
num_smoothing_copies=10

python main.py \
    --results_dir ./results \
    --target_model vicuna \
    --attack NoDefensePrompt \
    --attack_logfile data/GCG/vicuna_behaviors.json \
    --defense NoPertub