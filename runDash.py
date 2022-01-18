#!/usr/bin/env python3

import dash
from dash import dcc,html
import dash_daq as daq
from dash.dependencies import Output, Input
import plotly.graph_objects as go
import uproot
from HitsAndTracksPlotter import HitsAndTracksPlotter
import os
import argparse
import socket

hit_options_ = ["RecHitHGC", "SimHitMuonCSC", "SimHitPixelECLowTof", "SimHitPixelLowTof",
                    "SimHitHGCEE", "SimHitHGCHEF", "SimHitHGCHEB", ]

particle_options_ = ["GenPart", "CaloPart", "PFTruthPart", "Track", "TrackDisp", "TrackingPart", "PFCand", "None", ]
particle_coptions_ = ["Index", "PFTruthPartIdx", "pdgId",]
simcluster_options_ = ["SimCluster", "MergedSimCluster", "MergedByDRSimCluster"]
hit_colors_ = [x+"Idx" for x in simcluster_options_]+\
            ["SimClusterIdx", "CaloPartIdx", "pdgId", "PFCandIdx", "PFTruthPartIdx", "PFTICLCandIdx"]
default_dataset_ = "testGun_nano.root"
dataset = default_dataset_
base_path = os.path.expanduser("~/cernbox") if "macbook" in socket.gethostname() else "/eos/user/k/kelong"
ntuple_path = f"{base_path}/ML4Reco/Ntuples"
globalplotter = HitsAndTracksPlotter(f"{ntuple_path}/{dataset}")

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default=default_dataset_, type=str, help="Input file")
    parser.add_argument("-p", "--particles", default="Track", type=str, choices=particle_options_)
    parser.add_argument("-s", "--simClusters", default="None", type=str, choices=simcluster_options_)
    parser.add_argument("-hc", "--hitColor", default="pdgId", type=str, choices=hit_colors_)
    parser.add_argument("-c", "--particleColor", default="Index", type=str, choices=particle_coptions_)
    parser.add_argument("-e", "--event", default=1, type=int, help="Event number to show")
    parsers = parser.add_subparsers(dest='mode')
    interactive = parsers.add_parser("interactive", help="Launch and interactive dash session")
    output = parsers.add_parser("output", help="Produce plots as output (not interactive)")
    output.add_argument("-o", "--outputFile", default="event_display", type=str, help="Output file")
    output.add_argument("--outDir", default="plots/", type=str, help="Output plots directory")
    output.add_argument("-np", "--pileupAsNoise", action='store_true', help="Color pileup as noise")
    output.add_argument("-tc", "--trackPtCut", default=1, type=float, help="Min pt for particles/tracks")
    return parser.parse_args()
 
args = parseArgs()

def draw_plots(hitTypes, detectors, pileupAsNoise, colormode, pcolormode, particles, simclusters, event, nHitFilter, trackPtCut, dataset):
    if not dataset:
        dataset = args.dataset
    if not event:
        event = 0
    plotter = globalplotter
    plotter.setSimClusters(simcluster_options_)
    plotter.setSimClusterHitFilter(nHitFilter if nHitFilter else 0)
    plotter.setHits(hitTypes)
    plotter.setPileupAsNoise(pileupAsNoise)
    if event != plotter.getEvent() or dataset not in plotter.getDataset():
        plotter.setEvent(event)
        fulldataset = dataset if os.path.isfile(dataset) else f"{ntuple_path}/{dataset}"
        plotter.setDataset(fulldataset)
        plotter.setReload()
    plotter.setDetectors(detectors)
    plotter.setTrackPtCut(trackPtCut)
    if particles != "None":
        plotter.setParticles(particles)
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
        min=0, max=100, step=1,
    ),
    html.Br(),
    html.Label('Data set'),
    dcc.Dropdown(
        id='dataset',
        options=[
            {'label': "10 particle gun (no PU)", 'value': "partGun10_noPU_nano.root"},
            {'label' : '10 particle gun (w/ PU)', 'value' : "partGun10_pileup100_nano.root"},
            {'label': "single photon gun (w/ PU)", 'value': "PhotonNoTrackerGun1cmOffsetOldMerging_nano.root"},
            {'label' : 'single photon gun (no PU)', 'value' : "PhotonNoTrackerGun1cmOffsetOldMerging_noPU_nano.root"},
            {'label' : 'TTbar (fineCalo)', 'value' : "TTbar_fineCalo_nano.root"},
            {'label': 'Merging (fineCalo) Default','value':"Gun50Part_CHEPDef_fineCalo_nano_default.root"}
        ],
        value=args.dataset
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
        value=args.particles,
    ),
    html.Label('Hit color mode'),
    dcc.Dropdown(
        id='colormode',
        options=[{'label': i, 'value': i} for i in hit_colors_],
        value=args.hitColor,
    ),
    html.Label('Particle color mode'),
    dcc.Dropdown(
        id='pcolormode',
        options=[{'label': i, 'value': i} for i in particle_coptions_],
        value=args.particleColor,
    ),
    html.Label('Show SimClusters'),
    dcc.Dropdown(
        id='simclusters',
        options=[ {'label' : i, 'value' : i} for i in simcluster_options_],
        value=args.simClusters
    ),
    html.Br(),
    html.Label('Filter SimClusters by nHits'),
    html.Br(),
    dcc.Input(
        id="nHitFilter", type="number", placeholder="minHits", debounce=True,
        min=0, max=20, step=1,
    ),
    html.Br(),
    html.Label('Minimum particle/track pt'),
    html.Br(),
    dcc.Input(
        id="trackPtCut", type="number", placeholder="ptCut", value=1.,
    ),
    daq.BooleanSwitch('pileupAsNoise', label="Pileup as noise", labelPosition="top", on=True),
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
    [Input("pileupAsNoise", "on")],
    [Input("colormode", "value")],
    [Input("pcolormode", "value")],
    [Input("particles", "value")],
    [Input("simclusters", "value")],
    [Input("event", "value")],
    [Input("nHitFilter", "value")],
    [Input("trackPtCut", "value")],
    [Input("dataset", "value")],
)


def draw_figure(hitTypes, detectors, pileupAsNoise, colormode, pcolormode, particles, simclusters, event, nHitFilter, trackPtCut, dataset):
    return draw_plots(hitTypes, detectors, pileupAsNoise, colormode, pcolormode, particles, simclusters, event, nHitFilter, trackPtCut, dataset)

if __name__ == '__main__':
   if args.mode == "interactive":
	   app.run_server(debug=True, port=3389, host='0.0.0.0')
   elif args.mode == 'output':
      static_plot_opts = {'hitTypes':['RecHitHGC'],
                   'detectors':[],
                   'colormode': args.hitColor,
                   'pcolormode': args.particleColor, 
                   'particles': args.particles,
                   'simclusters': args.simClusters,
                   'event':args.event,
                   'nHitFilter':20, 
                   'dataset':args.dataset,
                   'pileupAsNoise' : args.pileupAsNoise,
                   'trackPtCut' : args.trackPtCut,
      }
      fig = go.Figure(draw_plots(**static_plot_opts))
      if not os.path.exists(args.outDir):
          os.makedirs(args.outDir) 
      outputFileName = args.outDir+'/' + args.outputFile+'_event_'+str(args.event)+'.html'
      fig.write_html(outputFileName)
   else:
      raise ValueError("Must select mode 'interactive' or 'output'")
