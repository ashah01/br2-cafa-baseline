#!/bin/bash

python cafa_evals.py \
    --input_dir "/home/adibvafa/BioReason2/evals/v7.8B.10" \
    --ontology "../data/go-basic.obo" \
    --ia_file "../data/IA.txt" \
    --output_dir "eval_results/v7.8B.10" \
    --threads 0
