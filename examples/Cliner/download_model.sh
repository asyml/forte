#!/bin/bash
# download the pre-trained model
wget https://drive.google.com/file/d/1Jlm2wdmNA-GotTWF60zZRUs1MbnzYox2 -O CliNER/models/train_full.model

# download the dependency package for evaluation
wget https://drive.google.com/file/d/1ZVgJ7EQtMjPpg_v-lCycCLdFgVzmdTxI -O CliNER/tools/i2b2va-eval.jar