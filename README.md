# Meaning\_Alone

Let-Alone Scalar Semantics Experiments





* Generate probabilities for each model: predict.sh
* Generate Probabilities for each model checkpoint: predict\_chkpts.sh (in progress)



* Evaluate predictions: eval.sh (in progress)



## Overview

This repository holds code for evaluating learning trajectories of scalar constructions. Code is available for both gathering log-probabilities from checkpoints and evaluation. 



## Dataset

* combined\_easy.tsv is currently the main dataset
* Columns: 



## Set Up

* pip install -r requirements.txt



## Running Experiments: Predictions

* Run predict\_pythia or predict\_ettin to query the checkpoints that are specified within. Edit the files to query different checkpoints.
* Predictions are stored in the outputs directory, with subdirectories for the model and subsubdirectories for the checkpoints
* output files are same as input but with extra columns for log probabilities.
* Affinity scores are stored in last columns (null values for decoders).



## Running Experiments: Evaluation

* eval\_accuracy.sh to get accuracy of ettin checkpoints (in progress)

## List of Scripts:
* predict.py: used to generate log-probabilities / affinity scores for a PairedFocus cxn dataset.
* predict_blimp.py: generate log-probabilities for BLiMP.
* predict_comps.py: generate log-probabilites for COMPS. 
* predict_ewok.py: generate log-probabilities for EWoK.
