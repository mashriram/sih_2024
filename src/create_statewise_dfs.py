import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# SOURCE_DIR = os.environ.get("DATA_SOURCE_DIR", "../sih_2024_data_source/")
# STATE_WISE_DATA = SOURCE_DIR + "statewise_results/"


# state_datasets: dict[str, pd.DataFrame] = dict()


# def create_statewise_datasets():
#     print(os.listdir(STATE_WISE_DATA))
#     files = [
#         os.path.join(STATE_WISE_DATA, file)
#         for file in os.listdir(STATE_WISE_DATA)
#         if file.endswith(".csv")
#     ]
#     for file in files:
#         print(file)
#         state_name = file.split("/")[-1].lower().replace("_all_years.csv", "")
#         print(state_name)
#         df = pd.read_csv(file)
#         df.rename(
#             columns={
#                 "1": "district_name",
#                 "2": "market_name",
#                 "3": "commodity",
#                 "4": "variety",
#                 "5": "grade",
#                 "6": "min_rs_quintal",
#                 "7": "max_rs_quintal",
#                 "8": "modal_rs_quintal",
#                 "9": "date",
#             },
#             inplace=True,
#         )
#         df.drop(columns="0", inplace=True)
#         df["year"] = df.date.str.split(" ").str[2]
#         df["month"] = df.date.str.split(" ").str[1]
#         df["day_of_month"] = df.date.str.split(" ").str[0]
#         state_datasets[state_name] = df

#     return state_datasets


# def get_statewise_datasets():
#     if len(state_datasets) == 0:
#         create_statewise_datasets()
#     return state_datasets


def get_statewise_datasets(
    states: list[str],
    commodities: list[str],
    start_year: int,
    end_year: int,
):
    """
    Retrieves data for specified states, commodities, and years in a Just-in-Time manner.

    Args:
        states (list): List of state names (lowercase) to retrieve data for.
        commodities (list): List of commodity names to retrieve data for.
        start_year (int): Starting year for data retrieval.
        end_year (int): Ending year for data retrieval.

    Returns:
        dict: A dictionary containing state names as keys and DataFrames of filtered data as values.
    """

    SOURCE_DIR = os.environ.get("DATA_SOURCE_DIR", "../sih_2024_data_source/")
    STATE_WISE_DATA = SOURCE_DIR + "statewise_results/"

    state_datasets = dict()
    for state in states:
        state_file = os.path.join(STATE_WISE_DATA, f"{state}_all_years.csv")
        if not os.path.exists(state_file):
            print(f"Data file not found for state: {state}")
            continue

        try:
            df = pd.read_csv(state_file)
        except (FileNotFoundError, pd.errors.ParserError) as e:
            print(f"Error reading data file for state {state}: {e}")
            continue
        # Data transformation (consider renaming and dropping columns if needed)
        df.rename(
            columns={
                "1": "district_name",
                "2": "market_name",
                "3": "commodity",
                "4": "variety",
                "5": "grade",
                "6": "min_rs_quintal",
                "7": "max_rs_quintal",
                "8": "modal_rs_quintal",
                "9": "date",
            },
            inplace=True,
        )
        df.drop(columns="0", inplace=True)
        df["year"] = df.date.str.split(" ").str[2]
        df["month"] = df.date.str.split(" ").str[1]
        df["day_of_month"] = df.date.str.split(" ").str[0]

        # Efficient filtering for specific years
        df = df[
            (df["year"].astype(int) >= start_year)
            & (df["year"].astype(int) <= end_year)
        ]

        # Efficient filtering for specific commodities (if provided)
        if commodities:
            df = df[df["commodity"].str.lower().isin(commodities)]
        state_datasets[state] = df

    return state_datasets


def to_csv_all_frames(datasets: dict[str, pd.DataFrame], path: str, name: str):
    try:
        os.listdir(path)
    except FileNotFoundError:
        os.mkdir(path)
    for state in datasets.keys():
        datasets[state].to_csv(path + "/" + state + name + ".csv")


state_datasets = get_statewise_datasets(
    states=["WB", "UP", "MP", "RJ", "PB"],
    commodities=["rice", "wheat", "potato", "onion"],
    start_year=2018,
    end_year=2024,
)
to_csv_all_frames(state_datasets, "./data/5_yr_data", "5_years")
