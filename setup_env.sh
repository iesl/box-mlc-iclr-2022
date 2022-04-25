#!/bin/bash
echo "=======Creating virtual env========="
virtualenv -p `which python3` .venv_multilabel_learning
source .venv_multilabel_learning/bin/activate

echo "=======Install test requirements======="
pip install -r test_requirements.txt

echo "=======Install doc requirements======="
pip install -r doc_requirements.txt

echo "=======Install core requirements======"
pip install -r core_requirements.txt

<<<<<<< HEAD
echo "=======Login to wandb (optional)==============="
wandb init
