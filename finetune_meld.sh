#!/bin/bash
 
python finetune_meld.py --data_dir MELD.Raw --task multitask --learning_rate 3e-5 --avg_checkpoints 10 --output_dir meld_finetuned_avg
