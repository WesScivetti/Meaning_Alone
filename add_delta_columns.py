from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


def add_delta_columns(input_file: str, output_file: str, target_col: str = "accuracy"):
    """
    Adds delta columns to the input TSV file and writes the updated DataFrame to the output TSV file.

    Args:
        input_file: Path to input TSV file with a target column, either score or accuracy.
        output_file: Path to output TSV file to write the updated DataFrame.
        target_col: Target column name.
    """


    df = pd.read_csv(input_file, sep='\t')

    current_scores = {}

    score_scaler = StandardScaler()

    #steps = sorted(df["revision"].unique())
    #cxns = df["construction"].unique()
    for r in df.index:
        rev = df.loc[r, "revision"]
        cxn = df.loc[r, "construction"]
        try:
            test_type = df.loc[r, "test_type"]
        except:
            try:
                test_type = df.loc[r, "world_knowledge_filter"]
            except:
                raise ValueError("No test type column found")
        score = df.loc[r, target_col]
        tup = (cxn, test_type)
        if tup not in current_scores:
            current_scores[tup] = score
            df.at[r, "delta_accuracy"] = 0
        else:
            delta = score - current_scores[tup]
            df.at[r, "delta_accuracy"] = delta
            current_scores[tup] = score


    df["delta_accuracy_zscale"] = score_scaler.fit_transform(df[["delta_accuracy"]])

    for r in df.index:
        rev = df.loc[r, "revision"]
        if rev == "step1":
            df.at[r, "delta_accuracy_zscale"] = 0


    df.to_csv(output_file, sep='\t', index=False)
    return df

def correlation_analysis(df: pd.DataFrame, pair1, pair2):
    """
    Computes the correlation between two columns in the DataFrame.

    Args:
        df: Input DataFrame.
        col1: First column name.
        col2: Second column name.

    Returns:
        Correlation coefficient between the two columns.
    """
    cxn, test_type = pair1
    cxn2, test_type2 = pair2
    try:
        df_pair1 = df[(df["construction"] == cxn) & (df["test_type"] == test_type)]
    except:
        try:
            df_pair1 = df[(df["construction"] == cxn) & (df["world_knowledge_filter"] == test_type)]
        except:
            raise ValueError("No test type column found")

    try:
        df_pair2 = df[(df["construction"] == cxn2) & (df["test_type"] == test_type2)]
    except:
        try:
            df_pair2 = df[(df["construction"] == cxn2) & (df["world_knowledge_filter"] == test_type2)]
        except:
            raise ValueError("No test type column found")

    #print(len(df_pair1["delta_accuracy_zscale"]))
    #print(len(df_pair2["delta_accuracy_zscale"]))
    corr_coefficient, p_value = spearmanr(df_pair1["delta_accuracy_zscale"], df_pair2["delta_accuracy_zscale"])

    #corr = df_pair1["delta_accuracy_zscale"].corr(df_pair2["delta_accuracy_zscale"])
    print("Correlation between", pair1, "and", pair2, "is:", corr_coefficient)
    return corr_coefficient



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True, help="Path to input TSV file")
    parser.add_argument("--output_tsv", type=str, required=True, help="Path to output TSV file")
    args = parser.parse_args()

    df = add_delta_columns(args.input_tsv, args.output_tsv)
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "all"))
    correlation_analysis(df, ("nevermind", "aligned"), ("ewok", "all"))
    correlation_analysis(df, ("letalone", "aligned"), ("comps", "all"))
    correlation_analysis(df, ("letalone", "aligned"), ("blimp", "all"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_physical-interactions"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_physical-dynamics"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_physical-relations"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_social-interactions"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_quantitative-properties"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_material-dynamics"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_spatial-relations"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_material-properties"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_agent-properties"))
    correlation_analysis(df, ("letalone", "aligned"), ("ewok", "domain_social-relations"))