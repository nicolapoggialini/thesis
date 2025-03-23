# Multi-Domain Conversational Agents based on Semantic Parsing and Retrieval of External Knowledge

This repository contains my thesis work, the code produced, and the final presentation. Repositories used to create the final model (such as SPRING and AMRSim) are not reported in here.

## Instructions to reproduce the training
In order to start the training process of the standard model, use the Python script 'wandb_train.py' present in './src/model' .
To train the AMR-augmented model, use the Python script 'wandb_train_amr.py' in the same folder.

All the experiments can be reproduced by changing the settings in the training files.

Before starting the training, it's necessary to create the knowledge base using the code in './src/amr' and './src/knowledge_base'.
