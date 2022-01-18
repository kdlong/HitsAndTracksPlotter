import uproot
import plotly.graph_objects as go
import matplotlib.cm
import matplotlib._color_data as mcd
import matplotlib.colors
import numpy as np
import pandas as pd
import math
import random
import colorsys
random.seed(0)
colors_ = [matplotlib.colors.to_rgb(x) for x in mcd.XKCD_COLORS.values()]
random.shuffle(colors_)

class HitsAndTracksPlotter(object):
    def __init__(self):
        self.__init__("")

    def __init__(self, rtfile):
        self.reload = True
        self.data = {}
        self.hits = []
        self.tree = None
        self.simClusters = []
        self.detectors = []
        self.particles = []
        self.event = 0
        self.scHitFilter = 0
        self.rtfile = rtfile
        self.xIsZ = True
        self.trackPtCut = 1.
        self.removePU = True
        self.endcap = 'all'
        self.hitCommonBranches = ["x", "y", "z", "energy",]
        self.hitBranches = {"SimHitHGCEE" : self.hitCommonBranches + ["SimClusterIdx"],
                "SimHitHGCHEF" : self.hitCommonBranches + ["SimClusterIdx"],
                "SimHitHGCHEB" : self.hitCommonBranches + ["SimClusterIdx"],
                "SimHitMuonCSC" : self.hitCommonBranches + ["pdgId"],
                "SimHitPixelECLowTof" : self.hitCommonBranches + ["pdgId"],
                "SimHitPixelLowTof" : self.hitCommonBranches + ["pdgId"],
                "RecHitHGC" : self.hitCommonBranches + \
                        ["PFCandIdx", "BestSimClusterIdx", "BestMergedSimClusterIdx", ]
                        + ["BestMergedByDRSimClusterIdx", "BestPFTruthPartIdx"],
                        }
        self.scBranches = ["impactPoint_x", "impactPoint_y", "impactPoint_z", "pdgId", 
                "nHits", "boundaryEnergy", "isTrainable", "onHGCFrontFace", "eventId", "bunchCrossing"]
        self.scAddBranches = {"SimCluster" : ["MergedSimClusterIdx", "CaloPartIdx", "recEnergy"],
                "MergedSimCluster" : ["recEnergy"],
                }
        candBranchesNoVtx = ["pt", "eta", "phi", "charge", "pdgId"]
        vtx = ["Vtx_x", "Vtx_y", "Vtx_z"]
        candBranches = candBranchesNoVtx + vtx
        trackBranches = candBranchesNoVtx[:-1]+vtx+["PFTruthPartIdx"]
        puBranches = ["eventId", "bunchCrossing"]
        self.particleBranches = {"Track" : trackBranches+["TrackingPartNumMatch"], 
            "TrackMatch" : ["TrackingPart_MatchIdx"],
            "GenPart" : candBranchesNoVtx,
            "TrackingPart" : trackBranches+["pdgId"]+puBranches,
        }
        self.particleBranches["CaloPart"] = self.particleBranches["GenPart"]+puBranches
        self.particleBranches["PFTruthPart"] = self.particleBranches["TrackingPart"]
        # Objects that have their own vertices
        self.vertices = ["TrackingPart", "PFCand", "Track", "TrackDisp"]

        cmap = matplotlib.cm.tab20
        self.all_colors = [cmap(i) for i in range(20)]
        random.shuffle(self.all_colors)
        # For a large number fall back to brute force
        self.all_colors.extend(colors_)
        # Set the preferred colors for specific pdgIds
        self.pdgIdsMap = {111 : cmap(18), 211 : cmap(0), 11 : cmap(4), 13 : cmap(3), 
                # kaons
                311 : cmap(9), 321 : cmap(9), 130 : cmap(9), 310 : cmap(9),
                22 : cmap(6), 2112 : cmap(11), 2212 : cmap(10),
                }

    def addHits(self, hitType):
        self.hits.append(hitType)

    def setHits(self, hitTypes):
        self.hits = hitTypes

    def setTrackPtCut(self, trackPtCut):
        self.trackPtCut = trackPtCut

    def setPileupAsNoise(self, pileupAsNoise):
        self.removePU = pileupAsNoise

    def setXIsZ(self, xIsZ):
        self.xIsZ = xIsZ

    def setDataset(self, dataset):
        self.rtfile = dataset

    def setEvent(self, event):
        self.event = event

    def setReload(self):
        self.reload = True

    def getEvent(self):
        return self.event

    def getDataset(self):
        return self.rtfile

    def setSimClusters(self, scs):
        self.simClusters = scs

    def setSimClusterHitFilter(self, nhits):
        self.scHitFilter = nhits

    def setDetectors(self, detectors):
        self.detectors = detectors

    def setParticles(self, particles):
        if type(particles) == list or type(particles) == tuple:
            self.particles = particles
        elif type(particles) == str:
            if particles in ["Track", "TrackDisp"]:
                self.particles = [particles, "TrackingPart", "TrackMatch"]
            else:
                self.particles = [particles]

    def makeDataFrame(self, label, branches, datatype="", obsname=""):
        if obsname == "":
            obsname = label
        if not self.reload and (label in self.data or (datatype in self.data and label in self.data[datatype])):
            return self.data[label] if label in self.data else self.data[datatype][label]

        if not self.tree:
            f = uproot.open(self.rtfile)
            self.tree = f["Events"]
        columns = ["_".join([obsname, b]) for b in branches]
        columns.append("event")
        active_columns = filter(lambda x: x in self.tree, columns)
        df = self.tree.arrays(active_columns, entry_start=self.event, entry_stop=self.event+1, library="pd")
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(self.event, level="entry")
        return df 

    def loadDataNano(self):
        self.data["hits"] = {h : self.makeDataFrame(h, self.hitBranches[h], "hits") for h in self.hits}
        self.data["simClusters"] = {s : self.makeDataFrame(s, 
            self.scBranches+(self.scAddBranches[s] if s in self.scAddBranches else []), "simClusters") for s in self.simClusters}
        self.simClusterPos = "impactPoint" 
        if self.data["simClusters"] and not hasattr(self.data["simClusters"]["SimCluster"], "SimCluster_%s_x" % self.simClusterPos):
            self.simClusterPos = "lastPos"

        if self.particles:
            # The replace is a bit of a hack, the TrackMatch just exists because the matching index has different size than others
            self.data["particles"] = {p: self.makeDataFrame(p, self.particleBranches[p], "particles", 
                                            obsname=(p if p != "TrackMatch" else "Track")) for p in self.particles}
            self.data["GenVtx"] = self.makeDataFrame("GenVtx", ["x", "y", "z"], "GenVtx")
        self.reload = False

    def simClusterIdx(self, hitlabel):
        scdf = self.data["simClusters"]["SimCluster"] if "SimCluster" in self.data["simClusters"] else None
        hitdf = self.data["hits"][hitlabel]
        if "SimHit" in hitlabel:
            return hitdf[hitlabel+"_SimClusterIdx"].to_numpy()
        else:
            return hitdf[hitlabel+"_BestSimClusterIdx"].to_numpy()

    def hitColors(self, label, colortype):
        colors = None
        hitdf = self.data["hits"][label]
        index = 0
        if "HitHGC" not in label:
            # Color by PDG ID only option for Muon/Tracker simhits
            colors = hitdf[label+"_pdgId"] if label+"_pdgId" in hitdf else [-1]
        else:
            idx = self.simClusterIdx(label)
            unmatched = idx < 0
            idx = np.where(~unmatched, idx, 0)
            scdf = self.data["simClusters"]["SimCluster"] if "SimCluster" in self.data["simClusters"] else None
            if colortype == "SimClusterIdx":
                colors = idx
            elif colortype == "pdgId":
                colors = scdf["SimCluster_pdgId"].to_numpy()[idx] if scdf is not None else [-1]
            elif colortype == "CaloPartIdx":
                colors = scdf["SimCluster_CaloPartIdx"].to_numpy()[idx] if scdf is not None else [-1]
            elif "Idx" in colortype:
                if "Sim" in label and colortype == "MergedSimClusterIdx":
                    colors = scdf["SimCluster_MergedSimClusterIdx"].to_numpy()[idx] if scdf is not None else [-1]
                else:
                    colors = hitdf[f"{label}_Best{colortype}"].to_numpy() if hitdf is not None else [-1]
                idx = colors
            else:
                raise ValueError("Invalid hit color choice %s" % colortype)
            if scdf is not None and self.removePU:
                ispu = (scdf["SimCluster_eventId"][idx] != 0) | (scdf["SimCluster_bunchCrossing"][idx] != 0)
                colors[ispu] = -2
             
            # Take -1, not the last index for properties accessed via SimCluster
            colors = np.where(~unmatched, colors, -1)

        return self.mapColors(colors)

    def hitPdgIds(self, hitlabel):
        hitdf = self.data["hits"][hitlabel]
        scdf = self.data["simClusters"]["SimCluster"] if "SimCluster" in self.data["simClusters"] else None
        if "HGC" not in hitlabel:
            return hitdf[hitlabel+"_pdgId"]
        scIdx = self.simClusterIdx(hitlabel)
        return scdf["SimCluster_pdgId"].to_numpy()[scIdx] if scdf is not None else np.full(len(scIdx), -1)

    def caloPartIdx(self, hitlabel):
        hitdf = self.data["hits"][hitlabel]
        scdf = self.data["simClusters"]["SimCluster"] if "SimCluster" in self.data["simClusters"] else None
        if "HGC" not in hitlabel:
            return -1
        scIdx = self.simClusterIdx(hitlabel)
        return scdf["SimCluster_CaloPartIdx"].to_numpy()[scIdx] if scdf is not None else [-1]

    def drawHits(self, label, colortype):
        df = self.data["hits"][label]

        x = df[label+('_x' if not self.xIsZ else '_z')]
        y = df[label+('_y' if not self.xIsZ else '_x')]
        z = df[label+('_z' if not self.xIsZ else '_y')]

        color = self.hitColors(label, colortype)
        # Should recycle this better
        pids = self.hitPdgIds(label)

        # Just to select a few clusters
        #filt = (caloidx == 18) | (caloidx == 34) | (caloidx == 33) | (caloidx == 46) | (caloidx == 35)
        filt = np.ones(len(x), dtype='bool')
        if self.endcap == '+':
            filt = z > 0
        elif self.endcap == '-':
            filt = z < 0
        x = x[filt]
        y = y[filt]
        z = z[filt]
        pids = pids[filt]
        color = color[filt]

        #Would like to add the merged simCluster index to Hit print info
        text = ["SimTrack pdgId: %i" % pid for pid in pids]
        if 'RecHitHGC' in label and label+"_BestPFTruthPartIdx" in df: 
            text = ["%s<br>PFTruthPartIdx: %i<br>MergedSimClusterIdx: %i" % (t, p, m) \
                    for t,p,m in zip(text, df[label+"_BestPFTruthPartIdx"],df[label+'_BestMergedSimClusterIdx'])]

        return go.Scatter3d(x=x, y=y, z=z,
                    mode='markers', 
                    marker=dict(line=dict(width=0), size=self.hitSize(label), 
                        color=color, 
                    ),
                    hovertemplate="x: %{y:0.2f}<br>y: %{z:0.2f}<br>z: %{x:0.2f}<br>%{text}<br>",
                    name=label, 
                    text=text,
        )

    def hitSize(self, label):
        df = self.data["hits"][label]
        energy = df[label+'_energy']
        sclabel = label+("_SimClusterIdx" if "RecHit" not in label else "_BestSimClusterIdx")
        isnoise = df[sclabel] == -1
        scale = 16/np.average(energy[~isnoise])
        maxsize = 30
        minnotnoise = 2
        loge = np.log(energy*scale)
        loge[loge > maxsize] = maxsize
        loge[loge < 0] = 0
        loge[loge < 1 & ~isnoise] = minnotnoise
        #vals = 2*np.ones_like(energy)
        #vals[~isnoise] = 0.25
        return loge
    
    def drawAllHits(self, colortype):
        return [self.drawHits(hits, colortype) for hits in self.hits]

    def simClusterDrawText(self, label):
        # TODO make more transparent and configurable
        #    df = self.data["simClusters"][label] 
        pos = label+"_"+self.simClusterPos
        df = self.data["simClusters"][label]
        # This is efectively just an all true condition
        filt = df[label+"_nHits"] > self.scHitFilter
        #if label+'_isTrainable' in df:
        #    filt = df[label+'_isTrainable'] & df[label+'_onHGCFrontFace']
        df_filt = df[filt]
        scidx = df_filt.index

        recLabel = label+"_recEnergy"
        recEnergy = np.zeros(len(df)) if recLabel not in df_filt else df_filt[recLabel]
        text = ["Idx: %i<br>nHits: %i<br>pdgId: %i<br>energy: %0.2f: recEnergy: %.2f" % (i,n,p,e,r) for (i,n,p,e,r) \
                    in zip(scidx, df_filt[label+"_nHits"], df_filt[label+"_pdgId"], df_filt[label+"_boundaryEnergy"], recEnergy)]

        #TODO: Clean up
        if label == "MergedSimCluster":
            unmergedLabel = []
            unmerged = self.data["simClusters"]["SimCluster"]
            for i in scidx:
                entry = unmerged[unmerged["SimCluster_MergedSimClusterIdx"] == i].index
                unmergedLabel.append("; ".join(["-".join([str(j) for j in i]) for i in makeRanges(entry)]))
            text = ["%s<br>Unmerged Idxs: %s" % (t,u) for t,u in zip(text, unmergedLabel)]
        return text

    def drawSimClusters(self, label):
        if not label or label == "None":
            return []

        df = self.data["simClusters"][label]
        pos = label+"_"+self.simClusterPos
        if label+'_isTrainable' in df:
            filt = df[label+'_isTrainable'] & df[label+'_onHGCFrontFace']
        filt = df[label+"_nHits"] > self.scHitFilter
        df_filt = df[filt]
        
        text = self.simClusterDrawText(label)

        return [go.Scatter3d(x = df_filt[pos+'_z'], y = df_filt[pos+'_x'], z = df_filt[pos+'_y'],
                    mode='markers',
                    marker=dict(line=dict(width=1, color='DarkSlateGrey', ),
                        symbol='x', 
                        size=2, 
                        color=self.mapColors(df_filt.index), 
                    ),
                    hovertemplate="x: %{y:0.2f}<br>y: %{z:0.2f}<br>z: %{x:0.2f}<br>%{text}<br>",
                    name=label, text=text,
                )
        ]

    def PtEtaPhiVectors(self, label):
        df = self.data["particles"][label]
        pt = df[label+"_pt"]
        eta = df[label+"_eta"]
        phi = df[label+"_phi"]
        return np.stack((pt, eta, phi), axis=-1)

    def momentumVectors(self, label):
        df = self.data["particles"][label]
        pt = df[label+"_pt"]
        eta = df[label+"_eta"]
        phi = df[label+"_phi"]
        return np.stack((pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)), axis=-1)

    def makeVertices(self, label):
        df = self.data["particles"][label] if label in self.vertices else self.data["GenVtx"]
        vtxlabel = label+"_Vtx" if label in self.vertices else "GenVtx"
        vert = np.stack((df[vtxlabel+"_x"], df[vtxlabel+"_y"], df[vtxlabel+"_z"]), axis=-1)
        if vert.shape[0] < self.data["particles"][label].shape[0]:
             vert = np.resize(vert, (self.data["particles"][label].shape[0], vert.shape[1])) 
        return vert

    def trajectoryEndPoint(self, label):
        df = self.data["particles"][label]
        ids = df[label+"_pdgId"] if label+"pdgId" in df else np.zeros_like(df[label+"_eta"])
        eta = df[label+"_eta"]
        end = np.array([1000 if abs(i) == 13 else 350 for i in ids])
        end = end*np.sign(eta)
        decayz = np.full(len(ids), 1000) if label+"DecayVtx_z" not in df else df[label+"_DecayVtx_z"]
        filt = np.abs(decayz) < np.abs(end)
        end[filt] = decayz[filt]
        return end

    def drawParticles(self, label, colortype):
        print("Drawing particles", label)
        if not self.particles or "particles" not in self.data or not label in self.data["particles"]:
            return []
        df = self.data["particles"][label]
        mom = self.momentumVectors(label)
        vtx = self.makeVertices(label)
        charge = self.data["particles"][label][label+"_charge"]
        ids = df[label+"_pdgId"] if label+"_pdgId" in df else np.zeros_like(charge)
        if colortype == "PFTruthPartIdx":
            ids = df[label+"_PFTruthPartIdx"]
        end = self.trajectoryEndPoint(label)
        # Should make this array based
        ptEtaPhi = self.PtEtaPhiVectors(label)
        traces = []
        isPU = np.zeros(len(charge), dtype=int)
        if label in ["Track", "TrackDisp"] and all([x in self.data["particles"] for x in ["TrackMatch", "TrackingPart"]]):
            tpdf = self.data["particles"]["TrackingPart"]
            tpisPU = (tpdf["TrackingPart_eventId"] != 0) | (tpdf["TrackingPart_bunchCrossing"] != 0)
            tpid = tpdf["TrackingPart_pdgId"]
            bestmatch = np.cumsum(df["Track_TrackingPartNumMatch"])-df["Track_TrackingPartNumMatch"][0]
            matchdf = self.data["particles"]["TrackMatch"]
            if "Track_TrackingPart_MatchIdx" in matchdf:
                tpidx = np.where(bestmatch >= 0, matchdf["Track_TrackingPart_MatchIdx"][bestmatch], -1)
                ids = np.where(tpidx >= 0, tpid[tpidx], -1)
                isPU = np.where(tpidx >= 0, tpisPU[tpidx], -1)
        elif label == "CaloPart":
            isPU = (df["CaloPart_eventId"] != 0) | (df["CaloPart_bunchCrossing"] != 0)

        for i, (v,m,e,q) in enumerate(zip(vtx, mom, end, charge)):
            pt,eta,phi = ptEtaPhi[i]
            if pt < self.trackPtCut:
                continue
            pid = ids[i] 
            points = self.trajectory(v, m, e, q) 
            #lightenPileup = self.removePU and isPU[i]
            #color = self.mapColor(i if colortype == "Index" else pid, 0.25 if lightenPileup else 1.0)
            color = self.mapColor(i if colortype == "Index" else pid)
            if self.removePU and isPU[i]:
               color = self.mapColor(-2, 1.)

            idtext = 'pdgId' if colortype != 'PFTruthPartIdx' else "PFTruthPartIdx"
            text = [f'{idtext}: {pid}<br>isPU: {isPU[i]}<br>p<sub>T</sub>, η, phi:  ({pt:0.2f} GeV, {eta:0.2f}, {phi:0.2f})' 
                            for p in points]
            traces.append(go.Scatter3d(x=points[:,2], y=points[:,0], z=points[:,1],
                    mode='lines', name=f"{label}Idx{i} (pdgId={pid})", 
                    hovertemplate="x: %{y}<br>y: %{z}<br>z: %{x}<br>%{text}<br>",
                    text=text,
                    line=dict(color=color)
                )
            )
        return traces

    def mapColors(self, vals, alpha=1.):
        return np.array([self.mapColor(i, alpha) for i in vals])

    def mapColor(self, i, alpha=1.):
        if i == -1:
            return "#c8cbcc" 
        elif i == -2:
            return "#e1eef2"

        if abs(i) in self.pdgIdsMap:
            c = self.pdgIdsMap[abs(i)]
        else:
            c = self.all_colors[int(i) % len(self.all_colors)]

        if alpha != 1. and type(c) == tuple:
            if len(c) > 3:
                c = (*c[:3], alpha) 
            elif len(c) == 3:
                c = (*c, alpha)

        return matplotlib.colors.rgb2hex(c) 

    def makeLayout(self, uirev):
        # This can be done with camera, but it breaks uirevision
        xaxis = dict(range=[400, -400], title="x",
                    showgrid=True, gridcolor='white',
                    showbackground=True, backgroundcolor='#fafcff'
        )
        yaxis = dict(range=[-400, 400], title="y",
                    showgrid=True, gridcolor='white',
                    showbackground=True, backgroundcolor='#f7faff'
        )
        zaxis=dict(range=[1200,-1200], title="z (beamline)",
                showgrid=True, gridcolor='#aebacf',
                showbackground=True, backgroundcolor='#fafcff'
        )

        layout = dict(title="test",
                        scene = dict(
                            aspectmode='data',
                            aspectratio=dict(x=3 if self.xIsZ else 1,y=1,z=3 if not self.xIsZ else 1),
                            zaxis=zaxis if not self.xIsZ else yaxis,
                            yaxis=yaxis if not self.xIsZ else xaxis,
                            xaxis=xaxis if not self.xIsZ else zaxis,
                            # Broken for now
                            #camera = dict(
                            #    up=dict(x=0, y=1, z=0),
                            #),
                        ),
                        uirevision = uirev,
        )

        return layout

    def trajectory(self, initPos, initMom, endz, q):
        return self.neutralTrajectory(initPos, initMom, endz) if q == 0 else \
                    self.chargedTrajectory(initPos, initMom, endz, q) 

    def chargedTrajectory(self, initPos, initMom, endz, q):
        M0 = initPos
        P0 = initMom

        T0 = P0/np.linalg.norm(P0)
        H = np.array([0,0,1])

        s = (endz-M0[2])/T0[2]

        HcrossT = np.cross(H, T0)
        alpha = np.linalg.norm(HcrossT)
        N0 = HcrossT/np.linalg.norm(HcrossT)

        gamma = T0[2]
        Q = -3.8*2.99792458e-3*q/np.linalg.norm(P0)

        npoints = 100
        points = np.full((npoints,3), M0)
        step = np.linspace(M0, s, npoints)
        theta = Q*step
        M = M0[np.newaxis,:] + gamma*(theta-np.sin(theta))*H[np.newaxis,:]/Q \
                + np.sin(theta)*T0[np.newaxis,:]/Q + alpha*(1.-np.cos(theta))*N0/Q
        outsideTracker = ((M[:,0]*M[:,0]+M[:,1]*M[:,1]) > 113*113) & (M[:,2] < 300)
        firstOutside = np.argmax(outsideTracker) if np.count_nonzero(outsideTracker) else npoints
        points[npoints-firstOutside:,:] = M[:firstOutside,:]
        return points

    def neutralTrajectory(self, initPos, initMom, endz):
        M0 = initPos
        P0 = initMom
        
        points = np.linspace(M0, (P0[0]/P0[1], P0[1]/P0[2], endz), 100)
        offTracker = (points[:,0]*points[:,0]+points[:,1]*points[:,1]) > 113*2
        return points[~offTracker,:]

    def drawAllObjects(self, colormode, pcolormode, simclusters):
        data = self.drawAllHits(colormode)
        for label in self.particles:
            # Tracking part only used for ID here
            if label != "TrackMatch" and not (label == "TrackingPart" and len(self.particles) > 1):
                data.extend(self.drawParticles(label, pcolormode))
        data.extend(self.drawSimClusters(simclusters))
        data.extend(self.drawDetectors())
        return data

    def drawDetectors(self):
        detectors = []
        if "CSC front" in self.detectors:
            detectors.extend(self.drawCSCME1())
        if "Tracker" in self.detectors:
            detectors.append(self.drawTracker())
        if "HGCAL front" in self.detectors:
            detectors.extend(self.drawHGCFront())
        return detectors

    def drawTracker(self):
        x, y, z = cylinder(113.5, 282*2, a=-282)
        return go.Surface(x=z, y=x, z=y,
                    colorscale = [[0, '#d7dff5'], [1, '#d7dff5']],
                    showscale=False,
                    name='Tracker',
                    hoverinfo='skip',
                    opacity=0.25)

    def drawCSCME1(self):
        x, y, z = boundary_circle(275, 580)
        return [go.Scatter3d(x=z, y=x, z=y,
                    mode='lines',
                    surfaceaxis=0,
                    line=dict(color='#f5ebd7'),
                    opacity=0.25,
                    hoverinfo='skip',
                    name='CSC ME1/1'),
                go.Scatter3d(x=-1*z, y=x, z=y,
                    mode='lines',
                    surfaceaxis=0,
                    line=dict(color='#f5ebd7'),
                    opacity=0.25,
                    hoverinfo='skip',
                    name='CSC ME-1/1'),
                ]

    def drawHGCFront(self):
        x, y, z = boundary_circle(125, 315)
        return [go.Scatter3d(x=z, y=x, z=y,
                    mode='lines',
                    surfaceaxis=0,
                    line=dict(color='#bacfbe'),
                    opacity=0.25,
                    name='HGCAL front',
                    hoverinfo='skip',
                    ),
                go.Scatter3d(x=-1*z, y=x, z=y,
                    mode='lines',
                    surfaceaxis=0,
                    line=dict(color='#bacfbe'),
                    opacity=0.25,
                    hoverinfo='skip',
                    name='HGCAL front'),
                ]

# From https://community.plotly.com/t/basic-3d-cylinders/27990
def cylinder(r, h, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(a, a+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

def boundary_circle(r, h, nt=100):
    """
    r - boundary circle radius
    h - height above xOy-plane where the circle is included
    returns the circle parameterization
    """
    theta = np.linspace(0, 2*np.pi, nt)
    x= r*np.cos(theta)
    y = r*np.sin(theta)
    z = h*np.ones(theta.shape)
    return x, y, z

def makeRanges(seq):
    if len(seq) < 2:
        return [seq]
    first = seq[0]
    result = []
    for i in range(1, len(seq)+1):
        if i == len(seq) or seq[i] != seq[i-1]+1:
            ins = [first, seq[i-1]] if seq[i-1] != first else [first]
            result.append(ins)
            if i < len(seq):
                first = seq[i]
    return result

