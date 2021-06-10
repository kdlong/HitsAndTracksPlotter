import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import uproot
from HitsAndTracksPlotter import HitsAndTracksPlotter
import os

app = dash.Dash(__name__)
hit_options_ = ["RecHitHGC", "SimHitMuonCSC", "SimHitPixelECLowTof", "SimHitPixelLowTof",
                    "SimHitHGCEE", "SimHitHGCHEF", "SimHitHGCHEB", ]
default_dataset_ = "Gun50Part_CHEPDef_fineCalo_treeMerger_nano.root"

app.layout = html.Div([
    dcc.Graph(id="scatter-plot", style={'width': '90%', 'height': '60%'}),
    dcc.Input(
        id="event", type="number", placeholder="event",
        min=0, max=10, step=1,
    ),
    html.Br(),
    html.Label('Data set'),
    dcc.Dropdown(
        id='dataset',
        options=[
            {'label': "50 particle gun (fineCalo)", 'value': "Gun50Part_CHEPDef_fineCalo_treeMerger_nano.root"},
            {'label' : '50 particle gun (fineCalo=Off)', 'value' : "Gun50Part_CHEPDef_fineCalo_treeMerger_nano.root"},
            {'label' : 'TTbar (fineCalo)', 'value' : "TTbar_fineCalo_nano.root"},
        ],
        value=default_dataset_
    ),
    html.Br(),
    html.Label('Hit types'),
    dcc.Checklist(
        id='hitTypes',
        options=[{'label': i, 'value': i} for i in hit_options_
            ],
        value=hit_options_[:1],
    ),
    html.Label('Draw detector'),
    dcc.Checklist(
        id='detectorElements',
        options=[{'label': i, 'value': i} for i in 
            ["Tracker", "CSC front", "HGCAL front"]],
        value=[],
    ),
    html.Label('Particle trajectories'),
    dcc.Dropdown(
        id='particles',
        options=[{'label': i, 'value': i} for i in 
            ["GenPart", "TrackingPart", "PFCand", "CaloPart", "None"]],
        value="CaloPart"
    ),
    html.Label('Hit color mode'),
    dcc.Dropdown(
        id='colormode',
        options=[{'label': i, 'value': i} for i in ["MergedSimClusterIdx", "MergedByDRSimClusterIdx", 
            "SimClusterIdx", "CaloPartIdx", "pdgId", "PFCandIdx", "PFTICLCandIdx"]],
        value='CaloPartIdx'
    ),
    html.Label('Particle color mode'),
    dcc.Dropdown(
        id='pcolormode',
        options=[{'label': i, 'value': i} for i in ["Index", "pdgId",]],
        value='Index'
    ),
    html.Label('Show SimClusters'),
    dcc.Dropdown(
        id='simclusters',
        options=[{'label': "Default", 'value': "SimCluster"}, 
            {'label' : "Merged", "value" : "MergedSimCluster"},
            {'label' : "MergedByDR", "value" : "MergedByDRSimCluster"},
            {'label' : "None", "value" : "None"}],
        value="None"
    ),
    html.Br(),
    html.Label('Filter SimClusters by nHits'),
    html.Br(),
    dcc.Input(
        id="nHitFilter", type="number", placeholder="minHits",
        min=0, max=20, step=1,
    ),
    html.Br(),
    ],
    style={
        "width": "100%",
        "height": "1800px",
        "display": "inline-block",
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
    [Input("nHitFilter", "value")],
    [Input("dataset", "value")],
)
def draw_figure(hitTypes, detectors, colormode, pcolormode, particles, simclusters, event, nHitFilter, dataset):
    if not dataset:
        dataset = default_dataset_
    if not event:
        event = 0
    # Merged by dR off for now
    #plotter.setSimClusters(["SimCluster", "MergedSimCluster", "MergedByDRSimCluster"])
    plotter = globalplotter
    plotter.setSimClusters(["SimCluster", "MergedSimCluster", "MergedByDRSimCluster"])
    plotter.setSimClusterHitFilter(nHitFilter if nHitFilter else 0)
    plotter.setHits(hitTypes)
    if event != plotter.getEvent() or dataset not in plotter.getDataset():
        plotter.setEvent(event)
        plotter.setDataset(f"{ntuple_path}/{dataset}")
        plotter.setReload()
    plotter.setDetectors(detectors)
    plotter.setParticles(particles if particles != "None" else None)
    globalplotter.loadDataNano()

    data = plotter.drawAllObjects(colormode, pcolormode, simclusters)
    return {
        # For now never reset the camera
        'layout' : plotter.makeLayout('alwaystrue'),
        'data' : data,
    }

dataset = default_dataset_
ntuple_path = os.path.expanduser("~/cernbox/ML4Reco/Ntuples")
print("Set plotter")
globalplotter = HitsAndTracksPlotter(f"{ntuple_path}/{dataset}")

if __name__ == '__main__':
	app.run_server(debug=True, port=3389, host='0.0.0.0')
