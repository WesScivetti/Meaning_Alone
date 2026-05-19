# Meaning\_Alone

## Overview

This is the repo for the paper: "Language Models Learn Constructional Semantics, *Not To Mention* Syntax: Investigating LM Understanding of Paired-Focus  Constructions", to be presented at CoNLL 2026. The scripts here will allow you to generate predictions for a paired-focus dataset, run paired-focus syntactic and semantic evaluations, and run additional evaluations (blimp, comps, ewok, affinity scores) for model checkpoints.

For convenience, several shell scripts are provided that loop through lists of models.

* Generate probabilities for each model: predict.sh
* Generate Probabilities for each model checkpoint: predict\_chkpts.sh (in progress)



* Evaluate predictions: eval.sh (in progress)







## Dataset

* combined\_easy.tsv is currently the main dataset
* Columns:



## Set Up

* pip install -r requirements.txt
* The python version that was used for these experiments was python 3.13.5



## Running Experiments: Predictions

* Run predict\_pythia or predict\_ettin to query the checkpoints that are specified within. Edit the files to query different checkpoints.
* Predictions are stored in the outputs directory, with subdirectories for the model and subsubdirectories for the checkpoints
* output files are same as input but with extra columns for log probabilities.
* Affinity scores are stored in last columns (null values for decoders).



## Running Experiments: Evaluation

* eval\_accuracy.sh to get accuracy of ettin checkpoints (in progress)

## List of Scripts:

* predict.py: used to generate log-probabilities / affinity scores for a PairedFocus cxn dataset.
* predict\_blimp.py: generate log-probabilities for BLiMP.
* predict\_comps.py: generate log-probabilites for COMPS.
* predict\_ewok.py: generate log-probabilities for EWoK.
