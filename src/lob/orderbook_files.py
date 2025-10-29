# =============================================================
# lob/orderbook_files.py
#
# Functions to load order book data files.
# =============================================================

import pandas as pd

def load_orderbook_data(paths):
    """
    Load multiple order book files and concatenate them.

    Parameters:

    paths : list[str]
        File paths to the order book day files (expected Feather format).

    Returns:

    pandas.DataFrame | None
        A single DataFrame containing the concatenated rows in the order
        the paths were provided, or `None` if no file could be read.

    Notes:
    
    Each file is attempted independently. If reading a file fails, the error is
    reported and that file is skipped; successfully read files are concatenated
    with `ignore_index=True`.
    """
    dataframes = []
    for path in paths:
        try:
            df = pd.read_feather(path)
            dataframes.append(df)
        except Exception as e:
            print(f"[lob.files] Could not read '{path}' ({type(e).__name__}: {e})")
    
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return None