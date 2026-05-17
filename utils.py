import pandas as pd

def combine_two_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combines two DataFrames, replacing rows in df1 with rows from df2 where "old_index" column in df2 matches the index of df1.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        on_col: Column name to merge on.

    Returns:
        A combined DataFrame with columns from both input DataFrames.
    """
    df1 = df1.copy()
    df2 = df2.copy()
    print("Combining DataFrames")
    for row in df2.index:
        old_index = df2.loc[row, "old_index"]
        df1.loc[old_index] = df2.loc[row].drop("old_index")
    print("Done")
    return df1

