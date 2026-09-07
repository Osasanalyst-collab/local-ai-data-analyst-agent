import pandas as pd


def read_csv_safely(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file using common encodings.

    UTF-8 is attempted first, followed by encodings commonly found
    in Windows-exported and legacy CSV files.
    """
    encodings = (
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
    )

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
            )

        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error

    return pd.read_csv(file_path)

    