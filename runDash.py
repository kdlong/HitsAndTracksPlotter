import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import uproot
from HitsAndTracksPlotter import HitsAndTracksPlotter

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="scatter-plot", style={'width': '100%', 'height': '70%'}),
    html.Label('HitTypes'),
    dcc.Checklist(
        id='hitTypes',
        options=[{'label': i, 'value': i} for i in 
            ["SimHitHGCEE", "SimHitHGCHEF", "SimHitHGCHEB"]],
        value=["SimHitHGCEE", "SimHitHGCHEF", "SimHitHGCHEB"],
    ),
    html.Label('ColorStyle'),
    dcc.Dropdown(
        id='colormode',
        options=[{'label': i, 'value': i} for i in ["MergedSimClusterIdx", "SimClusterIdx", "pdgId",]],
        value='pdgId'
    ),
    ],
    style={
        "width": "100%",
        "height": "1000px",
        "display": "inline-block",
        "border": "3px #5c5c5c solid",
        "padding-top": "5px",
        "padding-left": "1px",
        "overflow": "hidden"
    }
)

@app.callback(
    Output("scatter-plot", "figure"), 
    [Input("hitTypes", "value")],
    [Input("colormode", "value")],
)
def draw_figure(hitTypes, colormode):
    plotter = HitsAndTracksPlotter("/Users/kenneth/cernbox/ML4Reco/Ntuples/Gun10Part_CHEPDef_nanoNoFineCalo.root")
    plotter.setSimClusters(["SimCluster", "MergedSimCluster"])
    plotter.setHits(hitTypes)
    #print("Setting")
    #plotter.setSimClusters(["SimCluster", "MergedSimCluster"])
    plotter.loadDataNano()

    return {
        'layout' : plotter.makeLayout(),
        'data' : plotter.drawAllHits(colormode)
    }

if __name__ == '__main__':
	app.run_server(debug=True, port=3389, host='0.0.0.0')
