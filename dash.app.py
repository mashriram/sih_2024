import plotly.express as px
from dash import Dash, Input, Output, dcc, dependencies, html

from create_statewise_dfs import get_statewise_datasets

app = Dash(__name__)

andhra_onions = get_statewise_datasets()["ap"]

app.layout = html.Div(
    [
        html.H4("Analysis of Onion Data"),
        dcc.Dropdown(
            id="plot_type",
            options=["scatter", "bar"],
            value="scatter",
        ),
        html.Div(
            id="scatter-dropdown-container",
            children=[
                dcc.Dropdown(
                    id="scatter-multi-dropdown",
                    options=andhra_onions.columns,
                    value=[
                        "modal_rs_quintal",
                        "variety",
                    ],  # Exclude year initially (optional)
                    multi=True,
                )
            ],
            style={"display": "none"},  # Initially hide the dropdown
        ),
        html.Div(
            id="box-dropdown-container",
            children=[
                dcc.Dropdown(
                    id="box-dropdown",
                    options=andhra_onions.columns,
                    value=[
                        "modal_rs_quintal",
                    ],  # Exclude year initially (optional)
                )
            ],
            style={"display": "none"},  # Initially hide the dropdown
        ),
        dcc.Graph(id="graph"),
    ]
)


@app.callback(
    dependencies.Output("scatter-dropdown-container", "style"),
    [dependencies.Input("plot_type", "value")],
)
def update_scatter_dropdown_visibility(plot_type):
    if plot_type == "scatter":
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    dependencies.Output("box-dropdown-container", "style"),
    [dependencies.Input("plot_type", "value")],
)
def update_box_dropdown_visibility(plot_type):
    if plot_type == "bar":
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output("graph", "figure"),
    [
        Input("plot_type", "value"),
        Input("scatter-multi-dropdown", "value"),
        Input("box-dropdown", "value"),
    ],
)
def update_chart(plot_type, scatter_dims, box_dims):
    if plot_type == "scatter":
        color: str = ""
        if "variety" in scatter_dims:
            color = "variety"
        elif "district_name" in scatter_dims:
            color = "district_name"
        elif "grade" in scatter_dims:
            color = "grade"
        elif "month" in scatter_dims:
            color = "month"
        fig = px.scatter_matrix(
            andhra_onions[scatter_dims], color=color
        )  # Use scatter matrix for scatter plots
    elif plot_type == "bar":
        #
        fig = px.box(
            data_frame=andhra_onions,
            y=andhra_onions[box_dims],
            color="district_name",
        )
        # Cleanup temporary column (optional)
    else:
        # Handle other plot types (optional)
        fig = None
    return fig


if __name__ == "__main__":
    app.run(debug=True)
