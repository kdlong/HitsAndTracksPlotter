# HitsAndTracksPlotter
Interactive 3D visualization of CMS sim/reco using plotly

Relies on NanoHGCML ntuples. Instructions for generating these can be found [here](https://github.com/kdlong/production_tests)

The code is a restructured version of [this repository](https://github.com/kdlong/SimClusterVisualization) using [dash](https://plotly.com/dash/) to make an interactive dashboard.

Start the server with 

```bash
pip install -r requirements.txt
python3 runDash.py
``` 

Then direct your browser to localhost:3389
