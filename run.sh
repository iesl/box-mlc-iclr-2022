#!/bin/bash
model=mbm # or mvm mhm boxe chmcnn mbm-t mvm-t mhm-t
dataset=expr_fun # or expr_go spo_fun spo_go cellcycle_fun cellcycle_go derisi_fun derisi_go diatoms enron_corr imclef07a imclef07d
wandb_entity=box-mlc # replace with your own entity/username
wandb_project=test # replace with your own project
cuda_device=-1 # 0 for gpu
for seed in 123 874 3023 23 14 2942 12334 85 234 111
do
  echo Training ${model} on ${dataset} with seed ${seed}
  set -x
  allennlp train-with-wandb best_models_configs/reported-${dataset}_${model}-test/config.json \
  --include-package=box_mlc \
  --file-friendly-logging \
  --numpy_seed=${seed} --pytorch_seed=${seed} --random_seed=${seed} \
  --trainer.cuda_device="${cuda_device}" \
  --wandb-tags="dataset@${dataset},model@${model}" \
  --wandb-entity=${wandb_entity} \
  --wandb-project=${wandb_project}
  set +x
done