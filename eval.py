import pandas as pd
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
from utils import combine_two_dfs

def print_results_for_data_file(filename, cxn, log_probs=False, world_knowledge_filter=None, verb=None):
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
    mean_aligned_la = np.mean(XY_LetAlone_XY_scores)
    print("MEAN SCORE FOR SWAPPED STIMULUS",np.mean(XY_LetAlone_YX_scores))
    mean_la_rev_dir = np.mean(XY_LetAlone_YX_scores)
    print("MEAN SCORE FOR OR",np.mean(XY_Or_XY_scores))
    mean_aligned_or = np.mean(XY_Or_XY_scores)
    print("MEAN SCORE FOR SWAPPED STIMULUS AND OR",np.mean(XY_Or_YX_scores))
    mean_or_rev_dir = np.mean(XY_Or_YX_scores)
    print("Number of examples considered here", len(df.index)/2)
    print("\n")
    print("Now for the swapped condition")
    print("MEAN SCORE FOR LET-ALONE",np.mean(YX_LetAlone_XY_scores))
    mean_misaligned_la = np.mean(YX_LetAlone_XY_scores)
    print("MEAN SCORE FOR SWAPPED STIMULUS",np.mean(YX_LetAlone_YX_scores))
    mean_misaligned_la_rev_dir = np.mean(YX_LetAlone_YX_scores)
    print("MEAN SCORE FOR OR",np.mean(YX_Or_XY_scores))
    mean_misaligned_or = np.mean(YX_Or_XY_scores)
    print("MEAN SCORE FOR SWAPPED STIMULUS AND OR",np.mean(YX_Or_YX_scores))
    mean_misaligned_or_rev_dir = np.mean(YX_Or_YX_scores)
    print("Number of examples considered here", len(df.index)/2)
    return mean_aligned_la, mean_la_rev_dir, mean_aligned_or, mean_or_rev_dir, mean_misaligned_la, mean_misaligned_la_rev_dir, mean_misaligned_or, mean_misaligned_or_rev_dir

def print_results_for_data_file_accuracy(filename, cxn, log_probs=False, world_knowledge_filter="all", verb=None, print_output=True):
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

  if verb is not None:
      df = df[df["Plain Verb"] == verb]

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
      if world_knowledge_filter in ["all", "aligned", "aligned1"]:
          if logodds[0] > logodds[4]:
            correct += 1
          total += 1

      #Compare Row 1 and Row 5, if 5 has a greater logods accuracy +1
      if world_knowledge_filter in ["all", "aligned", "aligned2"]:
          if logodds[1] < logodds[5]: #1 should be less than 5
            correct += 1
          total += 1

      #Compare Row 2 and Row 6, 6 should have greater logodds
      if world_knowledge_filter in ["all", "misaligned", "misaligned1"]:
          if logodds[2] < logodds[6]: #2 should be less than 6
            correct += 1
          total += 1

      #Compare Row 3 and 7, 3 should have greater logodds
      if world_knowledge_filter in ["all", "misaligned", "misaligned2"]:
          if logodds[3] > logodds[7]: #3 should be greater than 7
            correct += 1
          total += 1

      count += 1

      #Reset Rows
      rows = []
  if print_output:
    print(f"Correct {correct} / {total} = {correct/total}")
  acc = correct/total
  return acc

def print_results_for_data_file_syntax_accuracy(full_df, cxn):
    """
    Syntactic eval for pf constructions
    """
    syn_df = full_df
    syn_df = syn_df[syn_df["Cxn"].isin([cxn, "and"])]
    #change "Score" column to be the negative of its original (surprisal values)
    syn_df["Score"] = -1 * syn_df["Score"]
    #chunk df into groups of 10 rows, each group of 10 rows corresponds to one example with 10 conditions (5 for and, 5 for cxn)
    corrects = defaultdict(int)
    totals = defaultdict(int)
    delta_surps = defaultdict(list)
    delta_surps_and = defaultdict(list)
    rows = []
    for r in syn_df.index:
        row = syn_df.loc[r]
        rows.append(row)
        if len(rows) == 12:
            #Compare the 6 rows for the cxn to the 6 rows for and
            cxn_rows = [r for r in rows if r["Cxn"] == cxn]
            and_rows = [r for r in rows if r["Cxn"] == "and"]


            #okay each of these have 6 rows
            #five differences each, score in row[0] to each of the others
            #then compare and to cxn
            s1 = cxn_rows[0].loc["Score"]
            s2 = cxn_rows[1].loc["Score"]
            s3 = cxn_rows[2].loc["Score"]
            s4 = cxn_rows[3].loc["Score"]
            s5 = cxn_rows[4].loc["Score"]
            s6 = cxn_rows[5].loc["Score"]

            and_s1 = and_rows[0].loc["Score"]
            and_s2 = and_rows[1].loc["Score"]
            and_s3 = and_rows[2].loc["Score"]
            and_s4 = and_rows[3].loc["Score"]
            and_s5 = and_rows[4].loc["Score"]
            and_s6 = and_rows[5].loc["Score"]

            #s2 - s1 is no NPI
            #we expect that NPIs make PFs ungrammatical. So s2 should be comparatively more surprising (lower score) than s1. Relative to and, the s2-s1 difference should be higher.
            diff_cxn = s2 - s1
            delta_surps["NPI"].append(diff_cxn)
            diff_and = and_s2 - and_s1
            delta_surps_and["NPI"].append(diff_and)
            if diff_cxn > diff_and:
                #print(r, "CORRECT", diff_cxn, diff_and)
                corrects["NPI"] += 1
            else:
                pass
                #print(r, "INCORRECT", diff_cxn, diff_and)
            totals["NPI"] += 1

            #psuedocleft is also ungrammatical
            diff_cxn = s3 - s1
            delta_surps["Pseudocleft"].append(diff_cxn)
            diff_and = and_s3 - and_s1
            delta_surps_and["Pseudocleft"].append(diff_and)
            if diff_cxn > diff_and:
                corrects["Pseudocleft"] += 1
            totals["Pseudocleft"] += 1

            #cp conjunction is also ungrammatical
            diff_cxn = s4 - s1
            delta_surps["CP Conjunction"].append(diff_cxn)
            diff_and = and_s4 - and_s1
            delta_surps_and["CP Conjunction"].append(diff_and)
            if diff_cxn > diff_and:
                corrects["CP Conjunction"] += 1
            totals["CP Conjunction"] += 1

            #VP conjunction should be a lot closer, but calculate accuracy the same way (should be closer to 50%)
            diff_cxn = s5 - s1
            delta_surps["VP Conjunction"].append(diff_cxn)
            diff_and = and_s5 - and_s1
            delta_surps_and["VP Conjunction"].append(diff_and)
            if diff_cxn > diff_and:
                corrects["VP Conjunction"] += 1
            totals["VP Conjunction"] += 1

            #gapped vp conjunction should be closer, but calculate accuracy the same way
            diff_cxn = s6 - s1
            delta_surps["VP Gap Conjunction"].append(diff_cxn)
            diff_and = and_s6 - and_s1
            delta_surps_and["VP Gap Conjunction"].append(diff_and)
            if diff_cxn > diff_and:
                corrects["VP Gap Conjunction"] += 1
            totals["VP Gap Conjunction"] += 1

        #reset rows
        if len(rows) == 12:
            rows = []


    scores = {}
    for key in corrects.keys():
        scores[key] = corrects[key] / totals[key]
        print(f"Syntactic accuracy for {key}: {scores[key]} ({corrects[key]} / {totals[key]})")

    for k in delta_surps.keys():
        print(f"Mean delta surprisal for {k}: {np.mean(delta_surps[k])} (cxn) vs {np.mean(delta_surps_and[k])} (and)")

    scores["Avg_Test"] = np.mean([scores["NPI"], scores["Pseudocleft"], scores["CP Conjunction"]])
    scores["Avg_Control"] = np.mean([scores["VP Conjunction"], scores["VP Gap Conjunction"]])

    delta_surps["Avg_Test"] = delta_surps["NPI"] + delta_surps["Pseudocleft"] + delta_surps["CP Conjunction"]
    delta_surps["Avg_Control"] = delta_surps["VP Conjunction"] + delta_surps["VP Gap Conjunction"]

    delta_surps_and["Avg_Test"] = delta_surps_and["NPI"] + delta_surps_and["Pseudocleft"] + delta_surps_and["CP Conjunction"]
    delta_surps_and["Avg_Control"] = delta_surps_and["VP Conjunction"] + delta_surps_and["VP Gap Conjunction"]

    return scores, delta_surps, delta_surps_and




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True, help="Input TSV file with results")
    parser.add_argument("--fixed_tsv", type=str, required=False, help="Input TSV file with results for syntactic eval")
    parser.add_argument("--construction", type=str, required=True, help="Construction to analyze")
    parser.add_argument("--log_probs", action="store_true", help="Whether to treat scores as log probabilities")
    parser.add_argument("--accuracy", action="store_true", help="Whether to compute accuracy instead of mean scores")
    parser.add_argument("--all_cxns", action="store_true", help="Whether to compute all Cxns")
    parser.add_argument("--syntax", action="store_true", help="Whether to compute syntactic accuracy")
    args = parser.parse_args()

    if args.syntax:
        syn_df = pd.read_csv(args.input_tsv, sep='\t')
        fixed_df = pd.read_csv(args.fixed_tsv, sep='\t')
        combined_df = combine_two_dfs(syn_df, fixed_df)
        print_results_for_data_file_syntax_accuracy(combined_df, args.construction)
        exit()

    reference_df = pd.read_csv(args.input_tsv, sep='\t')
    verbs = reference_df["Plain Verb"].unique().tolist()

    if not(args.all_cxns):
        if args.accuracy:
          print("ALL RESULTS")
          acc_full = print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs)
          print("ALIGNED RESULTS")
          acc_aligned = print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs, world_knowledge_filter="aligned")
          print("ALIGNED FORMATTED")
          print("MISALIGNED RESULTS")
          acc_misaligned = print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs, world_knowledge_filter="misaligned")

      #verb_accs = defaultdict(lambda: defaultdict(dict))
      # for v in verbs:
      #       print(f"Verb specific results for verb {v}")
      #       verb_accs[v]["all"] = print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs, verb=v)
      #       verb_accs[v]["aligned"] = print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs, world_knowledge_filter="aligned", verb=v)
      #       verb_accs[v]["misaligned"] = print_results_for_data_file_accuracy(args.input_tsv, args.construction, log_probs=args.log_probs, world_knowledge_filter="misaligned", verb=v)
        else:
          print_results_for_data_file(args.input_tsv, args.construction, log_probs=args.log_probs)

    else:
        if args.accuracy:
            accuracies = defaultdict(dict)
            for cxn in reference_df["Cxn"].unique().tolist():
                print(f"Results for construction {cxn}")
                acc_full = print_results_for_data_file_accuracy(args.input_tsv, cxn, log_probs=args.log_probs)
                accuracies[cxn]["full"] = acc_full
                print("ALIGNED RESULTS")
                acc_aligned = print_results_for_data_file_accuracy(args.input_tsv, cxn, log_probs=args.log_probs, world_knowledge_filter="aligned")
                accuracies[cxn]["aligned"] = acc_aligned
                print("MISALIGNED RESULTS")
                acc_misaligned = print_results_for_data_file_accuracy(args.input_tsv, cxn, log_probs=args.log_probs, world_knowledge_filter="misaligned")
                accuracies[cxn]["misaligned"] = acc_misaligned

