import uproot

class HitsAndTracksPlotter(object):
    def __init__():
        self.data = {}
    def loadDataNano(rtfile):
        uproot.open(rtfile)
        self.data["SimHitHGCEE"] = uproot["Events"]["SimHitHGCEE"


