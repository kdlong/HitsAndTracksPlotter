#!/usr/bin/env python3

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.graph_objects as go
import uproot
from HitsAndTracksPlotter import HitsAndTracksPlotter
import os
import argparse
import socket

hit_options_ = ["RecHitHGC", "SimHitMuonCSC", "SimHitPixelECLowTof", "SimHitPixelLowTof",
                    "SimHitHGCEE", "SimHitHGCHEF", "SimHitHGCHEB", ]

particle_options_ = ["GenPart", "CaloPart", "PFTruthPart", "Track", "TrackingPart", "PFCand", "None", ]
particle_coptions_ = ["Index", "PFTruthPartIdx", "pdgId",]
#default_dataset_ = "Gun50Part_CHEPDef_fineCalo_noProp_nano.root"
#default_dataset_ = "Gun2Tau_nano.root"
default_dataset_ = "temp_nano.root"
dataset = default_dataset_
base_path = os.path.expanduser("~/cernbox") if "macbook" in socket.gethostname() else "/eos/user/k/kelong"
ntuple_path = f"{base_path}/ML4Reco/Ntuples"
globalplotter = HitsAndTracksPlotter(f"{ntuple_path}/{dataset}")

def parseArgs():
    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers(dest='mode')
    interactive = parsers.add_parser("interactive", help="Launch and interactive dash session")
    output = parsers.add_parser("output", help="Produce plots as output (not interactive)")
    output.add_argument("-d", "--dataset", default=default_dataset_, type=str, help="Input file")
    output.add_argument("-e", "--event", default=1, type=int, help="Event number to show")
    output.add_argument("-o", "--outputFile", default="event_display", type=str, help="Output file")
    output.add_argument("--outDir", default="plots/", type=str, help="Output plots directory")
    output.add_argument("-p", "--particles", default="CaloPart", type=str, choices=particle_options_)
    output.add_argument("-c", "--particleColor", default="Index", type=str, choices=particle_coptions_)
    return parser.parse_args()
 

def draw_plots(hitTypes, detectors, colormode, pcolormode, particles, simclusters, event, nHitFilter, dataset):
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


app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="scatter-plot", style={'width': '90%', 'height': '60%'}),
    dcc.Input(
        id="event", type="number", placeholder="event",
        min=0, max=17, step=1,
    ),
    html.Br(),
    html.Label('Data set'),
    dcc.Dropdown(
        id='dataset',
        options=[
            {'label': "50 particle gun (fineCalo)", 'value': "Gun50Part_CHEPDef_fineCalo_treeMerger_nano.root"},
            {'label' : '50 particle gun (fineCalo=Off)', 'value' : "Gun50Part_CHEPDef_fineCalo_treeMerger_nano.root"},
            {'label' : 'TTbar (fineCalo)', 'value' : "TTbar_fineCalo_nano.root"},
            {'label': 'Merging (fineCalo) Default','value':"Gun50Part_CHEPDef_fineCalo_nano_default.root"}
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
        options=[{'label': i, 'value': i} for i in particle_options_],
        #value="CaloPart"
        value="Track"
    ),
    html.Label('Hit color mode'),
    dcc.Dropdown(
        id='colormode',
        options=[{'label': i, 'value': i} for i in ["MergedSimClusterIdx", "MergedByDRSimClusterIdx", 
            "SimClusterIdx", "CaloPartIdx", "pdgId", "PFCandIdx", "PFTruthPartIdx", "PFTICLCandIdx"]],
        value='CaloPartIdx'
    ),
    html.Label('Particle color mode'),
    dcc.Dropdown(
        id='pcolormode',
        options=[{'label': i, 'value': i} for i in particle_coptions_],
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
    return draw_plots(hitTypes, detectors, colormode, pcolormode, particles, simclusters, event, nHitFilter, dataset)


if __name__ == '__main__':
   args = parseArgs()
   if args.mode == "interactive":
	   app.run_server(debug=True, port=3389, host='0.0.0.0')
   elif args.mode == 'output':
      static_plot_opts = {'hitTypes':['RecHitHGC'],
                   'detectors':[],
                   'colormode': "CaloPartIdx",
                   'pcolormode': args.particleColor, 
                   'particles': args.particles,
                   'simclusters':'MergedSimCluster',
                   'event':args.event,
                   'nHitFilter':20, 
                   'dataset':args.dataset}
      fig = go.Figure(draw_plots(static_plot_opts['hitTypes'], static_plot_opts['detectors'], static_plot_opts['colormode'], static_plot_opts['pcolormode'], static_plot_opts['particles'], static_plot_opts['simclusters'], static_plot_opts['event'], static_plot_opts['nHitFilter'], static_plot_opts['dataset']))
      if not os.path.exists(args.outDir):
          os.makedirs(args.outDir) 
      outputFileName = args.outDir+'/' + args.outputFile+'_event_'+str(args.event)+'.html'
      fig.write_html(outputFileName)
   else:
      raise ValueError("Must select mode 'interactive' or 'output'")
