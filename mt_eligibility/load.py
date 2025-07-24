import os

import pandas as pd


def load_and_combine_csvs(csv_names: set = {"master", "qer", "annotations", "data_split"}) -> pd.DataFrame:
    # Load necessary CSV files
    csvs = []
    if "master" in csv_names:
        csvs.append(pd.read_csv(os.path.join(os.path.dirname(__file__), r"data/master.csv")))
    if "qer" in csv_names:
        csvs.append(pd.read_csv(os.path.join(os.path.dirname(__file__), r"data/qer.csv")))
    if "annotations" in csv_names:
        csvs.append(pd.read_csv(os.path.join(os.path.dirname(__file__), r"data/annotations.csv")))
    if "data_split" in csv_names:
        csvs.append(pd.read_csv(os.path.join(os.path.dirname(__file__), r"data/data_split.csv")))

    # If nothing is provided, return
    if not csvs:
        return

    # Join all CSVs on ID and return
    final = csvs[0]
    for csv in csvs[1:]:
        final = final.join(csv.set_index("ID"), on="ID", how="inner")

    return final.reset_index(drop=True)
