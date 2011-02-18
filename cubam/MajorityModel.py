from BinaryModel import *
from numpy.random import rand

class MajorityModel(BinaryModel):
    def __init__(self, filename=None):
        self.mdlPrm = {
            'addNoise' : False,
        }
        self.wkrIds = {}
        self.imgIds = {}
        if filename:
            self.load_data(filename)
        else:
            self._setup_prior()    
    
    def __del__(self):
        pass

    def load_data(self, filename, skipyaml=False):
        """
        Data is assumed to be in the format:
        imageId workerId label
        """
        # load the text data
        filein = open(filename)
        info = filein.readline().rstrip().split(' ')
        self.numLbls = int(info[2])
        self.numWkrs = int(info[1])
        self.numImgs = int(info[0])
        self.imgPrm = []
        for i in range(self.numImgs):
            self.imgPrm.append([0, 0]) # (frac +ve votes, total n votes)
        self.wkrLbls = dict((id, []) for id in range(self.numWkrs))
        self.imgLbls = dict((id, []) for id in range(self.numImgs))
        self.labels = []
        for (lino, line) in enumerate(filein):
            cols = [int(c) for c in line.rstrip().split(' ')]
            iId = cols[0]; wId = cols[1]; lij = int(cols[2]==1)
            self.wkrLbls[wId].append([iId, lij])
            self.imgLbls[iId].append([wId, lij])
            self.labels.append((iId, wId, lij))
            self.imgPrm[iId][0] += lij
            self.imgPrm[iId][1] += 1
        # renormalize img prm
        for i in range(len(self.imgPrm)):
            self.imgPrm[i][0] = float(self.imgPrm[i][0])/self.imgPrm[i][1]

    def get_num_wkrs(self):
        return self.numWkrs

    def get_num_imgs(self):
        return self.numImgs

    def get_num_lbls(self):
        return self.numLbls

    def set_model_param(self, raw=[], prm=None):
        """
        Sets model parameters.

        Arguments:
          - `raw`: raw parameter vector
          - `prm`: hash of model parameter values to be changed
        """
        if not prm is None:
            for (k, v) in prm.iteritems():
                self.mdlPrm[k] = v

    def set_worker_param(self, raw):
        pass

    def set_image_param(self, raw):
        self.imgPrm = [r for r in raw]

    def get_model_param(self):
        return {}

    def get_worker_param_raw(self):
        return {}

    def get_image_param_raw(self):
        return [p for p in self.imgPrm]

    def get_worker_param(self, id=None):
        return {}

    def get_image_param(self, id=None):
        return [p for p in self.imgPrm]

    def get_labels(self):
        if self.mdlPrm['addNoise']:
            return [int((self.imgPrm[i][0]+(rand()-.5)/self.imgPrm[i][1])>.5)\
                        for i in range(len(self.imgPrm))]
        else:
            return [int(self.imgPrm[i][0]>.5) for i \
                    in range(len(self.imgPrm))]

    # TODO: load and save parameters

    def optimize_worker_param(self):
        pass

    def optimize_image_param(self):
        pass

    def objective(self, prm=None):
        pass

    def image_objective(self, prm=None):
        pass
        
    def image_objective_range(self, imgId, prm):
        pass

    def worker_objective_range(self, wkrId, prm):
        pass

    def gradient(self, prm=None):
        return []

    def worker_gradient(self, prm=None):
        return []

    def image_gradient(self, prm=None):
        pass

    def get_num_wkr_lbls(self):
        return [len(self.wkrLbls[id]) for id in range(self.numWkrs)]

    def get_num_img_lbls(self):
        return [len(self.imgLbls[id]) for id in range(self.numImgs)]
