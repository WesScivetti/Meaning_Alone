import pandas as pd
from argparse import ArgumentParser


def eval_scalar_semantics(input_file: str):
    """
    Evaluates scalar semantics by computing the mean easier-harder scores for each construction from the input TSV file.

    Args:
        input_file: Path to input TSV file with columns 'Cxn' and 'easier-harder'.

    Returns:
        A dictionary with cxn as keys and their mean easier-harder scores as values.
    """
    df = pd.read_csv(input_file, sep='\t')

    df = df[df["Cxn"] == "or"]

    correct = 0
    total = 0
    for r in df.index:
        row = df.loc[r]
        swapped_cond = int(row.loc["Swapped_Condition"])
        swapped_stim = int(row.loc["Swapped_Stimulus"])
        let_alone = int(row.loc["Let-Alone"])

        logodds = row.loc["easier-harder"]

        if (swapped_cond, swapped_stim, let_alone) == (0, 0, 0):
            if logodds > 0:
                correct += 1
            
        total += 1



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True, help="Input TSV file with results")
