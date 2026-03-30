import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import base64
import io
import json

# ---------------------------------------------------
# App
# ---------------------------------------------------

app = dash.Dash(__name__)
server = app.server

# ---------------------------------------------------
# Layout
# ---------------------------------------------------

app.layout = html.Div(
    children=[

        html.H1(
            "📊 DMDO Post Dashboard",
            style={
                "textAlign": "center",
                "color": "white",
                "marginBottom": "30px"
            }
        ),

        # ---------------------------
        # Upload area
        # ---------------------------
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                "📂 Drag and Drop or ",
                html.A("Select CSV File")
            ]),
            style={
                "width": "100%",
                "height": "70px",
                "lineHeight": "70px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "textAlign": "center",
                "marginBottom": "30px",
                "backgroundColor": "rgba(255,255,255,0.05)",
                "color": "white",
                "borderColor": "#00d4ff"
            },
            multiple=False,
        ),

        # ---------------------------
        # Vertical dropdown controls
        # ---------------------------
        html.Div(
            children=[

                html.Label(
                    "X Axis",
                    style={"color": "white", "fontWeight": "bold"}
                ),
                dcc.Dropdown(
                    id="x-col",
                    options=[],
                    value=None,
                    placeholder="Select X column",
                    style={
                        "marginBottom": "20px",
                        "color": "black"
                    }
                ),

                html.Label(
                    "Y Axis",
                    style={"color": "white", "fontWeight": "bold"}
                ),
                dcc.Dropdown(
                    id="y-col",
                    options=[],
                    value=None,
                    placeholder="Select Y column",
                    style={
                        "marginBottom": "20px",
                        "color": "black"
                    }
                ),

                html.Label(
                    "Marker Size",
                    style={"color": "white", "fontWeight": "bold"}
                ),
                dcc.Dropdown(
                    id="size-col",
                    options=[],
                    value=None,
                    placeholder="Select size column",
                    style={
                        "marginBottom": "20px",
                        "color": "black"
                    }
                ),

            ],
            style={
                "maxWidth": "500px",
                "marginBottom": "30px"
            }
        ),

        # ---------------------------
        # Plot
        # ---------------------------
        dcc.Graph(
            id="xy-plot"
        ),

        # ---------------------------
        # Hidden data storage
        # ---------------------------
        dcc.Store(id="stored-data"),

    ],
    style={
        "fontFamily": "Segoe UI",
        "padding": "30px",
        "background": "linear-gradient(135deg,#0f2027,#203a43,#2c5364)",
        "minHeight": "100vh"
    }
)

# ---------------------------------------------------
# CSV Parser
# ---------------------------------------------------

def parse_contents(contents):

    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        df.columns = df.columns.str.strip()
    except Exception:
        return None

    return df

# ---------------------------------------------------
# Load uploaded data
# ---------------------------------------------------

@app.callback(
    Output("stored-data", "data"),
    Output("x-col", "options"),
    Output("y-col", "options"),
    Output("size-col", "options"),
    Output("x-col", "value"),
    Output("y-col", "value"),
    Input("upload-data", "contents"),
)
def load_data(contents):

    if contents is None:
        return None, [], [], [], None, None

    df = parse_contents(contents)

    if df is None:
        return None, [], [], [], None, None

    cols = [{"label": c, "value": c} for c in df.columns]

    default_x = df.columns[0]
    default_y = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    return (
        df.to_json(date_format="iso", orient="split"),
        cols,
        cols,
        cols,
        default_x,
        default_y
    )

# ---------------------------------------------------
# Plot callback
# ---------------------------------------------------

@app.callback(
    Output("xy-plot", "figure"),
    Input("stored-data", "data"),
    Input("x-col", "value"),
    Input("y-col", "value"),
    Input("size-col", "value"),
)
def update_plot(data_json, x_col, y_col, size_col):

    if data_json is None or x_col is None or y_col is None:
        return px.scatter(title="Upload CSV and select columns")

    if isinstance(data_json, str):
        data_json = json.loads(data_json)

    df = pd.DataFrame(
        data_json["data"],
        columns=data_json["columns"]
    )

    df.columns = df.columns.str.strip()

    # is_numeric_y = pd.api.types.is_numeric_dtype(df[y_col])

    if size_col:
        is_numeric_size = pd.api.types.is_numeric_dtype(df[size_col])
    else:
        is_numeric_size = False

    # ---------------------------------
    # Choose color column
    # ---------------------------------
    if size_col and is_numeric_size:
        color_col = size_col
    else:
        color_col = y_col

    is_numeric_color = pd.api.types.is_numeric_dtype(df[color_col])

    # ---------------------------------
    # Create scatter plot
    # ---------------------------------
    if is_numeric_color:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_continuous_scale="Turbo",
            template="plotly_dark"
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            template="plotly_dark"
        )

    # ---------------------------------
    # Marker size scaling
    # ---------------------------------
    if size_col and is_numeric_size:

        size_values = pd.to_numeric(df[size_col], errors="coerce")

        vmin = size_values.min()
        vmax = size_values.max()

        if vmax > vmin:
            sizes = 8 + (size_values - vmin) / (vmax - vmin) * 35
        else:
            sizes = 15

        fig.update_traces(
            marker=dict(
                size=sizes,
                line=dict(width=1, color="white")
            )
        )

    else:
        fig.update_traces(
            marker=dict(
                size=12,
                line=dict(width=1, color="white")
            )
        )

    # ---------------------------------
    # Layout
    # ---------------------------------
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        transition_duration=500
    )

    return fig

# ---------------------------------------------------
# Run
# ---------------------------------------------------

def main():
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)