import uproot
import plotly.graph_objects as go
import matplotlib.cm
import matplotlib._color_data as mcd
import numpy as np

class HitsAndTracksPlotter(object):
    def __init__(self):
        self.__init__("")

    def __init__(self, rtfile):
        self.data = {}
        self.hits = []
        self.simClusters = []
        self.event = 0
        self.rtfile = rtfile
        self.hitBranches = ["x", "y", "z", "energy", "SimClusterIdx"]
        self.scBranches = ["impactPoint_x", "impactPoint_y", "impactPoint_z", "pdgId", "MergedSimClusterIdx"]

        cmap = matplotlib.cm.get_cmap('tab20b')    
        # For a small number of clusters, make them pretty
        self.all_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
        # For a large number fall back to brute force
        self.all_colors.extend(list(mcd.XKCD_COLORS.values()))
        # Set the preferred colors for specific pdgIds
        self.pdgIdsMap = {1 : "#c8cbcc", 111 : "red", 211 : 'blue', 11 : 'green', 13 : 'orange', 
                                22 : "lightblue", 2112 : "pink", 2212 : "purple"}

    def addHits(self, hitType):
        self.hits.append(hitType)

    def setHits(self, hitTypes):
        self.hits = hitTypes

    def setEvent(self, event):
        self.event = event

    def setSimClusters(self, scs):
        self.simClusters = scs

    def makeDataFrame(self, label, branches):
        columns = ["_".join([label, b]) for b in branches if b != f"{label}Idx"]
        f = uproot.open(self.rtfile)
        events = f["Events"]
        return events.arrays(columns, entry_start=self.event, entry_stop=self.event+1)[0]

    def loadDataNano(self):
        uproot.open(self.rtfile)
        self.data["hits"] = {h : self.makeDataFrame(h, self.hitBranches) for h in self.hits}
        self.data["simClusters"] = {s : self.makeDataFrame(s, self.scBranches) for s in self.simClusters}

    def hitColors(self, label, colortype):
        scIdx = self.data["hits"][label][label+"_SimClusterIdx"]
        colors = None
        if colortype == "SimClusterIdx":
            colors = scIdx
        elif colortype in ["pdgId", "MergedSimClusterIdx"]:
            df = self.data["simClusters"]["SimCluster"] if "SimCluster" in self.data["simClusters"] else None
            if colortype == "pdgId":
                colors = df["SimCluster_pdgId"][scIdx]  if df is not None else [-1]
            elif colortype == "MergedSimClusterIdx":
                colors = df["SimCluster_MergedSimClusterIdx"][scIdx] if df is not None else [-1]
        else:
            raise ValueError("Invalid hit color choice %s" % colortype)
        return self.mapColors(colors)

    def drawHits(self, label, colortype, xIsZ=True):
        df = self.data["hits"][label]
        x = df[label+('_x' if not xIsZ else '_z')]
        y = df[label+('_y' if not xIsZ else '_x')]
        z = df[label+('_z' if not xIsZ else '_y')]
        
        color = self.hitColors(label, colortype)

        return go.Scatter3d(x=x, y=y, z=z,
                    mode='markers', 
                    marker=dict(line=dict(width=0), size=1, 
                        color=color, 
                    ),
                    name=label, 
        )
    
    def drawAllHits(self, colortype, xIsZ=True):
        return [self.drawHits(hits, colortype, xIsZ) for hits in self.hits]

    def mapColors(self, vals):
        print(vals)
        return [self.mapColor(i) for i in vals]

    def mapColor(self, i):
        i = int(i)
        if abs(i) in self.pdgIdsMap:
            return self.pdgIdsMap[abs(i)]
        if i < 0:
            return "#c8cbcc"
        if i >= len(self.all_colors):
            i = np.random.randint(0, len(self.all_colors))
        # Avoid too "smooth" of a transition for close by values
        idx = i if i % 2 else (20 if i < 20 else len(self.all_colors))-i-1
        return self.all_colors[idx]

    def makeLayout(self, xIsZ=True):
        # This can be done with camera, but it breaks uirevision
        xaxis = dict(range=[-400, 400], title="x",
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
                            aspectratio=dict(x=3 if xIsZ else 1,y=1,z=3 if not xIsZ else 1),
                            zaxis=zaxis if not xIsZ else yaxis,
                            yaxis=yaxis if not xIsZ else xaxis,
                            xaxis=xaxis if not xIsZ else zaxis,
                            # Broken for now
                            #camera = dict(
                            #    up=dict(x=0, y=1, z=0),
                            #),
                        ),
                        uirevision = 'alwaystrue',
        )

        return layout
