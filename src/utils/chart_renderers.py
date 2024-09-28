import pandas as pd
from streamlit_echarts import JsCode, st_echarts


def create_stacked_line_chart(df, by, column):

    by_cols = ["year"]
    by_cols.append(by)
    avg_price_year_commodity = round(df.groupby(by_cols)[column].agg("mean"))
    price_comm = avg_price_year_commodity.reset_index()
    print(price_comm, avg_price_year_commodity)
    years: list[str] = avg_price_year_commodity.index.levels[0].tolist()
    commodities: list[str] = avg_price_year_commodity.index.levels[1].tolist()
    print(commodities, years)

    series = [
        {
            "name": commodity,
            "type": "line",
            "stack": "Total",
            "data": price_comm.query(f"{by} == '{commodity}'")[f"{column}"].tolist(),
        }
        for year, commodity in zip(years, commodities)
    ]

    options = {
        "title": {"text": "Commodity Price over time"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": commodities},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": years,
        },
        "yAxis": {"type": "value"},
        "series": series,
    }

    st_echarts(options=options, height="400px")


def create_line_chart(df, column):
    years = sorted(list(df["year"].astype(str).unique()))
    print(years)

    avg_price_year = round(df.groupby(["year"])[column].agg("mean")).tolist()

    options = {
        "xAxis": {
            "type": "category",
            "data": years,
        },
        "yAxis": {"type": "value"},
        "series": [{"data": avg_price_year, "type": "line"}],
    }

    st_echarts(options=options)


def create_stacked_bar_chart(df, by: str, column):
    years = sorted(list(df["year"].astype(str).unique()))
    by_cols = ["year"]
    by_cols.append(by)
    avg_price_year_commodity = round(df.groupby(by_cols)[column].agg("mean"))

    commodities = avg_price_year_commodity.index.levels[1].tolist()
    print("By Cols", (commodities))

    price_comm = avg_price_year_commodity.reset_index()

    series = [
        {
            "name": commodity,
            "type": "bar",
            "stack": "Total",
            "label": {"show": True},
            "emphasis": {"focus": "series"},
            "data": price_comm.query(f"{by} == '{commodity}'")[f"{column}"].tolist(),
        }
        for year, commodity in zip(years, commodities)
    ]

    options = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": commodities},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": {"type": "value"},
        "yAxis": {
            "type": "category",
            "data": years,
        },
        "series": series,
    }
    st_echarts(options=options, height="500px")


def create_scatter_single_axis(df, commodity):
    years = sorted(list(df["year"].astype(str).unique()))
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    df = df.query(f"commodity == '{commodity}'")
    df["date"] = df["date"].str.replace(" ", "-")
    # Convert the 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"], format="mixed")

    # Add a new column for the day of the week
    df["day_of_week"] = df["date"].dt.day_name()

    price_year_month = df.groupby(["year", "month"])["modal_rs_quintal"].agg("mean")
    price_year_month_df = price_year_month.reset_index(name="avg_price")

    price_data = []
    year_idx = 0
    for year in years:
        month_idx = 0
        for month in months:
            price_month = []
            price_month.append(year_idx)
            price_month.append(month_idx)
            avg_month_price = price_year_month_df.query(
                f"year == {int(year)} and month == '{month}'"
            )["avg_price"].values
            if len(avg_month_price) == 0:
                price_month.append(0)
            else:
                price = round(float(avg_month_price[0]))
                price_month.append(price)
            price_data.append(price_month)
            month_idx += 1
        year_idx += 1

    # print("price_data", price_data)

    option = {
        "tooltip": {"position": "top"},
        "title": [
            {"textBaseline": "middle", "top": f"{(idx + 0.5) * 100 / 7}%", "text": year}
            for idx, year in enumerate(years)
        ],
        "singleAxis": [
            {
                "left": 150,
                "type": "category",
                "boundaryGap": False,
                "data": months,
                "top": f"{(idx * 100 / 7 + 5)}%",
                "height": f"{(100 / 7 - 10)}%",
                "axisLabel": {"interval": 2},
            }
            for idx, _ in enumerate(years)
        ],
        "series": [
            {
                "singleAxisIndex": idx,
                "coordinateSystem": "singleAxis",
                "type": "scatter",
                "data": [],
                "symbolSize": JsCode(
                    "function(data_item){return data_item[1]/60}"
                ).js_code,
            }
            for idx, _ in enumerate(years)
        ],
    }
    for data_item in price_data:
        option["series"][data_item[0]]["data"].append([data_item[1], data_item[2]])

    st_echarts(options=option, height="600px")
