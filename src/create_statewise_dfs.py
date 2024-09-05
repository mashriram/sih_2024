import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SOURCE_DIR = os.environ.get("DATA_SOURCE_DIR", "data/")
STATE_WISE_DATA = SOURCE_DIR + "statewise_results/Onion"


state_datasets: dict[str, pd.DataFrame] = dict()


def create_statewise_datasets():
    print(os.listdir(STATE_WISE_DATA))
    files = [
        os.path.join(STATE_WISE_DATA, file)
        for file in os.listdir(STATE_WISE_DATA)
        if file.endswith(".csv")
    ]
    for file in files:
        print(file)
        state_name = file.split("/")[-1].lower().replace("_all_years.csv", "")
        print(state_name)
        df = pd.read_csv(file)
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
        state_datasets[state_name] = df

    return state_datasets


def get_statewise_datasets():
    if len(state_datasets) == 0:
        create_statewise_datasets()
    return state_datasets
