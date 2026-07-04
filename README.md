# Meaning\_Alone

## Overview

This is the repo for the paper: "Language Models Learn Constructional Semantics, *Not To Mention* Syntax: Investigating LM Understanding of Paired-Focus  Constructions", to be presented at CoNLL 2026. The scripts here will allow you to generate predictions for a paired-focus dataset, run paired-focus syntactic and semantic evaluations, and run additional evaluations (blimp, comps, ewok, affinity scores) for model checkpoints.



## Set Up

* pip install -r requirements.txt
* The python version that was used for these experiments was python 3.13.5

## Dataset

* See data folder for details

## Preprocessing
* make_templates.py: used to generate example sentences templatically from scalar adjective set

## Running Experiments: Predictions

### Paired-Focus Semantics
* predict.py

### Paired-Focus Syntax
* predict_syntax.py 

### Other Datasets
* predict_blimp.py
* predict_comps.py
* predict_ewok.py

### Affinity
* affinity.py (don't need to run separately)

### Looping through Models


* Run predict\_pythia.sh or predict\_ettin.sh to query the checkpoints that are specified within. Edit the files to query different checkpoints.
* Predictions are stored in the outputs directory, with subdirectories for the model and subsubdirectories for the checkpoints
* output files are same as input but with extra columns for log probabilities.
* Affinity scores are stored in last columns (null values for decoders).



## Running Experiments: Evaluation

* eval.py
* eval_blimp.py
* evaluate_affinities.py
* main_eval.py

* eval\_accuracy.sh to get accuracy of loop checkpoints

## Correlation Analysis
* corr.py

## Visualizations (in progress)
