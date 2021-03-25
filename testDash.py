import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import uproot

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="scatter-plot", style={'width': '100%', 'height': '80%'}),
    html.Label('hitType'),
    dcc.Dropdown(
        id='hitType',
        options=[{'label': i, 'value': i} for i in ["SimHitHGCHEF", "SimHitHGCEE"]],
    ),
    html.Label('ColorStyle'),
    dcc.Dropdown(
        id='colormode',
        options=[{'label': i, 'value': i} for i in ["SimClusterIdx", "pdgId", "black"]],
    ),
    ],
    style={
        "width": "100%",
        "height": "800px",
        "display": "inline-block",
        "border": "3px #5c5c5c solid",
        "padding-top": "5px",
        "padding-left": "1px",
        "overflow": "hidden"
    }
)

def makeDataFrame(branches, rtfile, evt):
    f = uproot.open(rtfile)
    events = f["Events"]
    return events.arrays(branches, entry_start=evt, entry_stop=evt+1)[0]

def positionLabels(label):
    return ['_'.join([label, x]) for x in ["x", "y", "z"]]

def plotHits(df, label, color, xIsZ=True):
    x = df[label+('_x' if not xIsZ else '_z')]
    y = df[label+('_y' if not xIsZ else '_x')]
    z = df[label+('_z' if not xIsZ else '_y')]
    return go.Scatter3d(x=x, y=y, z=z,
                mode='markers', 
                marker=dict(line=dict(width=0), size=1, 
                    color=color, 
                    #color='black', 
                ),
                name="blah", 
                )

def makeLayout(xIsZ=True):
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

@app.callback(
    Output("scatter-plot", "figure"), 
    [Input("hitType", "value")],
    [Input("colormode", "value")],
)
def update_bar_chart(hitType, colormode):
    #mask = (df.petal_width > low) & (df.petal_width < high)
    print("Updating")
    if not hitType:
        hitType = "SimHitHGCEE"

    print("hitType is", hitType)
    hitsDf = makeDataFrame(positionLabels(hitType).append("%s_SimClusterIdx" % hitType), "/Users/kenneth/cernbox/ML4Reco/Ntuples/Gun10Part_CHEPDef_nanoNoFineCalo.root", 0)
    scDf = makeDataFrame(["SimCluster_pdgId"], "/Users/kenneth/cernbox/ML4Reco/Ntuples/Gun10Part_CHEPDef_nanoNoFineCalo.root", 0)
    print("colors are")
    colors = hitsDf[hitType+"_SimClusterIdx"]
    print("Here", type(colors))
    if colormode == "pdgId":
        colors = scDf["SimCluster_pdgId"][colors]
    elif colormode == "black":
        colors='black'
    #fig = go.Figure(data=[plotHits(df, hitType)], layout=makeLayout())
    # Just a temporary holder
    #fig.update_layout({"uirevision" : 'alwaystrue'})
    #fig["layout"]["uirevision"] = True
    return {
        'layout' : makeLayout(),
        #'layout' : dict(uirevision='alwaystrue',
        #                scene=dict(
        #                   aspectmode='manual',
        #                   aspectratio=dict(x=1,y=1,z=3),
        #                   # camera=dict(up=dict(x=0,y=1,z=0)),
        #                   xaxis=dict(range=[-400, 400], title="x",
        #                       showgrid=True, gridcolor='white', 
        #                       showbackground=True, backgroundcolor='#fafcff'),
        #                   yaxis=dict(range=[-400, 400], title="y", 
        #                       showgrid=True, gridcolor='white', 
        #                       showbackground=True, backgroundcolor='#f7faff'),
        #                   zaxis=dict(range=[-1600, 1600], title="z (beamline)",
        #                       showgrid=True, gridcolor='#aebacf', 
        #                       showbackground=True, backgroundcolor='#fafcff'),
        #                ),
        #            ),
        'data' : [plotHits(hitsDf, hitType, colors)],
    }

if __name__ == '__main__':
	app.run_server(debug=True, port=3389, host='0.0.0.0')
