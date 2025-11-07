import pandas as pd
import numpy as np
import re
from collections import defaultdict
from argparse import ArgumentParser

def print_results_for_data_file(filename, cxn, log_probs=False):
    """

    Args:
        filename:
        cxn:
        log_probs:

    Returns: Nothing, just prints to console. Prints the results for the given construction (cxn) from the data file.

    """
    old_df = pd.read_csv(filename, sep='\t')
    allowed_cxns = [cxn, "or"]
    df = old_df[old_df["Cxn"].isin(allowed_cxns)]
    #Okay just graph XY_LetAlone_XY, which is 0, 0, 1
    #Second point is XY_LetAlone_YX which is 0, 1, 1 and should lower the difference
    #First or point is XY_Or_XY which is 0, 0, 0 and should have some smaller positive difference than let-alone
    #Second or point is XY_Or_Yx which is 0, 1, 0 and should be kind of similar to the first Or point

    XY_LetAlone_XY_scores = [] #Easier should be quite probable
    XY_LetAlone_YX_scores = [] #Easier least probable of this bunch
    XY_Or_XY_scores = [] #Easier Probable, but less so than 1)
    XY_Or_YX_scores = [] #Easier less probable than 3) but more probable than 2)

    YX_LetAlone_XY_scores = [] #Smaller gap than 3)
    YX_LetAlone_YX_scores = [] #Here the Let-Alone is pushing to increase difference, against world knowledge. Stronger gap than 4)
    YX_Or_XY_scores = [] #Larger gap than 4
    YX_Or_YX_scores = [] #Harder should be probable but smaller difference than 2)

    for r in df.index:
        row = df.loc[r]
        condition_value = row.loc["Swapped_Condition"] #0 is XY, 1 is YX
        stimulus_value = row.loc["Swapped_Stimulus"] #0 is XY, 1 is YX
        let_alone_value = row.loc["Let-Alone"] #0 is Or, 1 is Let-Alone
        score = row.loc["easier-harder"]
        easier = row.loc["easier"]
        harder = row.loc["harder"]
        if log_probs:
          log_score = score
        else:
          log_score = np.log(easier / harder)

        if condition_value == 0 and stimulus_value == 0 and let_alone_value == 1: #0,0,1 Let-Alone
          XY_LetAlone_XY_scores.append(log_score)

        if condition_value == 0 and stimulus_value == 1 and let_alone_value == 1: #0,1,1 Swapped Stimulus Let-Alone
          XY_LetAlone_YX_scores.append(log_score)

        if condition_value == 0 and stimulus_value == 0 and let_alone_value == 0: #0,0,0 Or
          XY_Or_XY_scores.append(log_score)

        if condition_value == 0 and stimulus_value == 1 and let_alone_value == 0: #0,1,0 Swapped Stimulus Or
          XY_Or_YX_scores.append(log_score)

        if condition_value == 1 and stimulus_value == 0 and let_alone_value == 1: #1,0,1 Let-Alone
          YX_LetAlone_XY_scores.append(log_score)

        if condition_value == 1 and stimulus_value == 1 and let_alone_value == 1: #1,1,1 Swapped Stimulus Let-Alone
          YX_LetAlone_YX_scores.append(log_score)

        if condition_value == 1 and stimulus_value == 0 and let_alone_value == 0: #1,0,0 Or
          YX_Or_XY_scores.append(log_score)

        if condition_value == 1 and stimulus_value == 1 and let_alone_value == 0: #1,1,0 Swapped Stimulus Or
          YX_Or_YX_scores.append(log_score)



    print("MEAN SCORE FOR LET-ALONE",np.mean(XY_LetAlone_XY_scores))
    print("MEAN SCORE FOR SWAPPED STIMULUS",np.mean(XY_LetAlone_YX_scores))
    print("MEAN SCORE FOR OR",np.mean(XY_Or_XY_scores))
    print("MEAN SCORE FOR SWAPPED STIMULUS AND OR",np.mean(XY_Or_YX_scores))
    print("Number of examples considered here", len(df.index)/2)
    print("\n")
    print("Now for the swapped condition")
    print("MEAN SCORE FOR LET-ALONE",np.mean(YX_LetAlone_XY_scores))
    print("MEAN SCORE FOR SWAPPED STIMULUS",np.mean(YX_LetAlone_YX_scores))
    print("MEAN SCORE FOR OR",np.mean(YX_Or_XY_scores))
    print("MEAN SCORE FOR SWAPPED STIMULUS AND OR",np.mean(YX_Or_YX_scores))
    print("Number of examples considered here", len(df.index)/2)
    return

def print_results_for_data_file_accuracy(filename, cxn, log_probs=False, world_knowledge_filter="all"):
  """"
  This takes the input file and computes an accuracy score by pairwise comparing logodds
  0,0,1 should be greater than 0,0,0
  0,1,1 should be less than 0,1,0
  1,0,1 should be less than 1,0,0
  1,1,1 should be greater than 1,1,0
  """
  correct = 0
  total = 0
  old_df = pd.read_csv(filename, sep='\t')
  allowed_cxns = [cxn, "or"]
  df = old_df[old_df["Cxn"].isin(allowed_cxns)]
  score_dictionary = defaultdict(dict) #Will store the scores for [0,0][1] to correspond to 0,0,1 etc
  if world_knowledge_filter == "all":
      #don't filter anything
      pass

  # #NEEDS FIXING BECAUSE THE NUMBER OF ROWS HAS CHANGED
  # elif world_knowledge_filter == "aligned":
  #     #only compare 0,0,1 > 0,0,0 and 0,1,1 < 0,1,0
  #     df = df[df["Swapped_Condition"] == 0]
  #
  # elif world_knowledge_filter == "misaligned":
  #     #only compare 1,0,1 < 1,0,0 and 1,1,1 > 1,1,0
  #     df = df[df["Swapped_Condition"] == 1]

  count = 0
  rows = []
  for r in df.index:
    row = df.loc[r]
    rows.append(row)
    #print(rows)
    if len(rows) == 8:
      #print("HELLO")
      easiers = [r.loc["easier"] for r in rows]
      harders = [r.loc["harder"] for r in rows]
      if log_probs:
        logodds = [r.loc["easier-harder"] for r in rows]
      else:
        logodds = [np.log(e/h) for e,h in zip(easiers,harders)]

      #Compare Row 0 and 4, if 0 has greater logodds accuracy +1
      if world_knowledge_filter in ["all", "aligned"]:
          if logodds[0] > logodds[4]:
            correct += 1
          total += 1

      #Compare Row 1 and Row 5, if 5 has a greater logods accuracy +1
      if world_knowledge_filter in ["all", "aligned"]:
          if logodds[1] < logodds[5]: #1 should be less than 5
            correct += 1
          total += 1

      #Compare Row 2 and Row 6, 6 should have greater logodds
      if world_knowledge_filter in ["all", "misaligned"]:
          if logodds[2] < logodds[6]: #2 should be less than 6
            correct += 1
          total += 1

      #Compare Row 3 and 7, 3 should have greater logodds
      if world_knowledge_filter in ["all", "misaligned"]:
          if logodds[3] > logodds[7]: #3 should be greater than 7
            correct += 1
          total += 1

      count += 1

      #Reset Rows
      rows = []
  print(f"Correct {correct} / {total} = {correct/total}")
  return
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True, help="Input TSV file with results")
    parser.add_argument("--construction", type=str, required=True, help="Construction to analyze")
    parser.add_argument("--log_probs", action="store_true", help="Whether to treat scores as log probabilities")
    parser.add_argument("--accuracy", action="store_true", help="Whether to compute accuracy instead of mean scores")
    args = parser.parse_args()

    if args.accuracy:
      print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs)
      print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs, world_knowledge_filter="aligned")
      print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs, world_knowledge_filter="misaligned")
    else:
      print_results_for_data_file(args.input_tsv, args.construction, log_probs=args.log_probs)
