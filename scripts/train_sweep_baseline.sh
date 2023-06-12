#!/bin/bash
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph dataset_name=$1 models.net.num_layers=1 models.net.hidden_features=16 trainer.max_epochs=$2
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph dataset_name=$1 models.net.num_layers=1 models.net.hidden_features=32 trainer.max_epochs=$2
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph dataset_name=$1 models.net.num_layers=1 models.net.hidden_features=64 trainer.max_epochs=$2
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph dataset_name=$1 models.net.num_layers=2 models.net.hidden_features=16 trainer.max_epochs=$2
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph dataset_name=$1 models.net.num_layers=2 models.net.hidden_features=32 trainer.max_epochs=$2
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph dataset_name=$1 models.net.num_layers=2 models.net.hidden_features=64 trainer.max_epochs=$2
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph dataset_name=$1 models.net.num_layers=3 models.net.hidden_features=16 trainer.max_epochs=$2
PROJECT_ROOT=./ python src/train.py experiment=exp_resgated_multidigraph dataset_name=$1 models.net.num_layers=3 models.net.hidden_features=32 trainer.max_epochs=$2
