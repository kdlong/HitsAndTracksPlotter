import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import uproot
from HitsAndTracksPlotter import HitsAndTracksPlotter

app = dash.Dash(__name__)
hit_options_ = ["SimHitHGCEE", "SimHitHGCHEF", "SimHitHGCHEB", "SimHitMuonCSC", "SimHitPixelECLowTof", "SimHitPixelLowTof",
                    "RecHitHGCEE", "RecHitHGCHEF", "RecHitHGCHEB"]

app.layout = html.Div([
    dcc.Graph(id="scatter-plot", style={'width': '100%', 'height': '70%'}),
    dcc.Input(
        id="event", type="number", placeholder="event",
        min=0, max=10, step=1,
    ),
    html.Br(),
    html.Label('Particles per endcap'),
    dcc.Dropdown(
        id='numPart',
        options=[{'label': i, 'value': i} for i in 
            [10, 80]],
        value=10
    ),
    html.Br(),
    html.Label('Hit types'),
    dcc.Checklist(
        id='hitTypes',
        options=[{'label': i, 'value': i} for i in hit_options_
            ],
        value=hit_options_[:6],
    ),
    html.Label('Draw detector'),
    dcc.Checklist(
        id='detectorElements',
        options=[{'label': i, 'value': i} for i in 
            ["Tracker", "CSC front", "HGCAL front"]],
        value=["Tracker", "CSC front", "HGCAL front"],
    ),
    html.Label('Particle trajectories'),
    dcc.Dropdown(
        id='particles',
        options=[{'label': i, 'value': i} for i in 
            ["GenPart", "TrackingPart", "PFCand", "PFTICLCand", "CaloPart", "None"]],
        value="GenPart"
    ),
    html.Label('Hit color mode'),
    dcc.Dropdown(
        id='colormode',
        options=[{'label': i, 'value': i} for i in ["MergedSimClusterIdx", "SimClusterIdx", "CaloPartIdx", "pdgId",
            "PFCandIdx", "PFTICLCandIdx"]],
        value='pdgId'
    ),
    html.Label('Particle color mode'),
    dcc.Dropdown(
        id='pcolormode',
        options=[{'label': i, 'value': i} for i in ["Index", "pdgId",]],
        value='pdgId'
    ),
    html.Label('Show SimClusters'),
    dcc.Dropdown(
        id='simclusters',
        options=[{'label': "Default", 'value': "SimCluster"}, 
            {'label' : "Merged", "value" : "MergedSimCluster"},
            {'label' : "None", "value" : "None"}],
        value="None"
    ),
    ],
    style={
        "width": "100%",
        "height": "1600px",
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
    [Input("detectorElements", "value")],
    [Input("colormode", "value")],
    [Input("pcolormode", "value")],
    [Input("particles", "value")],
    [Input("simclusters", "value")],
    [Input("event", "value")],
    [Input("numPart", "value")],
)
def draw_figure(hitTypes, detectors, colormode, pcolormode, particles, simclusters, event, numPart):
    if numPart == 80:
        plotter = HitsAndTracksPlotter("/Users/kenneth/cernbox/ML4Reco/Ntuples/111_nanoML.root")
    else:
        plotter = HitsAndTracksPlotter("/Users/kenneth/cernbox/ML4Reco/Ntuples/Gun10Part_CHEPDef_fineCalo_nano.root")
    plotter.setSimClusters(["SimCluster", "MergedSimCluster"])
    plotter.setHits(hitTypes)
    plotter.setEvent(event if event else 0)
    plotter.setDetectors(detectors)
    if particles != "None":
        plotter.setParticles(particles)
    plotter.loadDataNano()

    data = plotter.drawAllObjects(colormode, pcolormode, simclusters)
    return {
        'layout' : plotter.makeLayout(numPart),
        'data' : data,
    }

if __name__ == '__main__':
	app.run_server(debug=True, port=3389, host='0.0.0.0')
