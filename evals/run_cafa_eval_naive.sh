#!/bin/bash

python cafa_evals_naive.py \
    --ontology "../bioreason2/dataset/go-basic.obo" \
    --ia-file "../data/IA.txt" \
    --output-dir "naive_results" \
    --cache-dir "cache" \
    --min-frequency 0.01 \
    --threshold-step 0.01 \
    --threads 0