# HitsAndTracksPlotter
Interactive 3D visualization of CMS sim/reco using plotly

Relies on NanoHGCML ntuples. Instructions for generating these can be found [here](https://github.com/kdlong/production_tests)

The code is a restructured version of [this repository](https://github.com/kdlong/SimClusterVisualization), now using [dash](https://plotly.com/dash/) to make an interactive dashboard.

# Setup
Setup the necessary pip environment with
```bash
pip install -r requirements.txt
```
Then ```. ./env/bin/activate``` before running (needed every time you open a new session)

Running is most convenient from your local computer, using lxplus etc you would need to play with tunneling or open the browser from lxplus. 

For this, you could copy output files locally using cernbox. The files in the central repo can be found at /eos/cms/store/user/kelong/ML4Reco/.

For example, the 50 particle sample can be copied as

```bash
cp /eos/cms/store/user/kelong/ML4Reco/Gun50Part_CHEPDef_NoPropagate/111_nanoML.root /eos/user/<yourcernbox>/ML4Reco/Ntuples
```

# Running

Two options are available : run an interactive dashboard (mode = interactive) or run with specific conditions and save event display in html format for a particular event (mode = output)

### Interactive dashboard
Launch the server with

```bash
python3 runDash.py interactive
``` 

and direct your browser to localhost:3389

### Run and save event display

```
python3 runDash.py output 
```
The following options are available : 
-d : input dataset
-e : event number to display
-o : output file
--outDir : output directory where to save the event display

# Use

This is a WIP, so you may find some that not all options work properly for all samples. Let me know if this is the case for you.

The goal is to visualize target clusters and tracks as well as the candidates we reconstruct. Several drop down menus and check boxes allow you to control what information is displayed.

**Particles per endcap** modifies the data set being used. In prinicple other samples could be produced, but the current options are generated with a particle Gun shot towards the HGCAL volume with 50 particles chosen randomly from muons, electrons, pions, and kaons.

**Hit types** selects which hits to display. Nothing prevents you from displaying both SimHits and RecHits, but it's a bad idea, because they will be on top of each other and things will run slower for no reason.

**Particle trajectories** show the trajectories of true particles or candidates. The trajectories are meant to guide you to the hits that should be collected for the recontruction, they are extrapolated based on the vertex, four vector and charge, and assuming a B = 3.8 T uniform field.

**Hit color mode** Color hits based 

* their PDG ID, based on association to SimTracks and the SimTrack pdgId
*  based on the candidate or SimCluster they are associated to. 
**NOTE**: Here MergedSimClusters are produced in the NanoAOD production step by merging SimClusters, produced with the fineCalo mode, based on deltaR

**Particle color mode** Color particles by index in the collection or by pdgId of the candidate. In the case of coloring by index, the hit index will match the particle index if the association is the same, e.g., by CaloParticles or by candidates

**Show SimClusters** Show the position of default SimClusters (produced in CMSSW) or the MergedSimClusters produced with the dR merging
