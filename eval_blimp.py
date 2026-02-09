import pandas as pd

def print_results_blimp_comps(filename, print_output=False):
    """
    Args:
        filename: should be something like outputs/model_name/revision/blimp_all.tsv

    Returns: blimp accuracy

    Coincidentally this works also with comps. Ewok has 4 sentences so is slightly different, see below.

    """

    df = pd.read_csv(filename, sep='\t')

    total = 0
    correct = 0

    for r in df.index:
        row = df.loc[r]
        good_log_prob = row.loc["log_prob_good"]
        bad_log_prob = row.loc["log_prob_bad"]

        total += 1
        if good_log_prob > bad_log_prob:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    if print_output:
        print(f"Results for file: {filename}")
        print(f"Total items: {total}")
        print(f"Correct items: {correct}")
        print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def print_results_ewok(filename, print_output=False, domain=None):
    """

    Args:
        filename: ewok_core.tsv
        print_output: if true, prints outputs to screen

    Returns: accuracy

    """
    df = pd.read_csv(filename, sep='\t')
    if domain:
        df = df[df["domain"] == domain]
    total = 0
    correct = 0
    for r in df.index:
        row = df.loc[r]
        # good_log_prob = row.loc["log_prob_good"]
        # bad_log_prob = row.loc["log_prob_bad"]

        total += 1

        #fields are: log_prob_t1_giv_c1, log_prob_t1_giv_c2, log_prob_t2_giv_c1, log_prob_t2_giv_c2
        #correctness is score(T1 | C1) > score(T1 | C2) and score(T2 | C1) < score(T2 | C2)
        if (row.loc["log_prob_t1_giv_c1"] > row.loc["log_prob_t1_giv_c2"]) and (row.loc["log_prob_t2_giv_c1"] < row.loc["log_prob_t2_giv_c2"]):
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    if print_output:
        print(f"Results for file: {filename}")
        print(f"Total items: {total}")
        print(f"Correct items: {correct}")
        print(f"Accuracy: {accuracy:.4f}")
    return accuracy
