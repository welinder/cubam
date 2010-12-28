from BinaryModel import *
from Binary1dSignalModel import Binary1dSignalModel
from numpy import sign, mod, sin, cos, dot
from utils import tw2tsw

class BinaryNdSignalModel(BinaryModel):
    def __init__(self, filename=None, dim=2):
        self._mdlPrmList = ['beta', 'sigx', 'sigw', 'muw', 'sigt', 'dim']
        BinaryModel.__init__(self, filename)
        self.set_model_param(prm={ 'dim' : dim })
    
    def get_model_param(self):
        prm = Model.get_model_param(self)
        prm['dim'] = int(prm['dim'])
        return prm

    def load_data(self, filename, skipyaml=False):
        Model.load_data(self, filename, skipyaml)
        self._dataFile = filename

    def image_objective_range(self, imgId, prm):
        pass
    
    def worker_objective_range(self, wkrId, prm=None, wj=None, tj=None):
        pass

    def get_worker_param(self, id=None):
        prm = self.get_worker_param_raw()
        dim = self.get_model_param()['dim']
        nwkrs = len(prm)/(1+dim)
        offset = nwkrs*dim
        if id is None:
            return dict((id, prm[dim*id:dim*(id+1)]+[prm[offset+id]]) \
              for id in range(nwkrs))
        else:
            return prm[dim*id:dim*(id+1)]+[prm[offset+id]]

    def get_image_param(self, id=None):
        prm = self.get_image_param_raw()
        dim = self.get_model_param()['dim']
        nimgs = len(prm)/dim
        if id is None:
            return dict((id, prm[dim*id:dim*(id+1)]) \
              for id in range(nimgs))
        else:
            return prm[dim*id:dim*(id+1)]

    def get_worker_var(self, wkrId):
        pass

    def get_image_var(self, imgId):
        pass

    def sample_worker_param(self, numWkr, tjPrior=0.8, angle=.3,
                            sigPrior=[1.5, .3], sigThresh=[.05, 3.],
                            tjThresh=[-1.5, 1.5], wjPrior=[2.0, 1.0]):
        """
        Sample worker parameters according to model.
        """
        dim = self.get_model_param()['dim']
        wkrPrm = {}
        for wkrId in range(numWkr):
            sj = gamma(sigPrior[0], sigPrior[1])
            # threshold the sj so we don't get too bad outliers
            sj = min([max([sj, sigThresh[0]]), sigThresh[1]])
            if dim==1:
                wj = [(-1. if (rand() < advPrior) else 1.)/sj]
            if dim==2:
                # sample angle centered at 45 degrees
                a = pi/4+randn()*angle
                wj = [sin(a)/sj, cos(a)/sj]
            else:
                wj = (randn(dim)*wjPrior[1]+wjPrior[0]).tolist()
            tj = randtn(*tjThresh)*tjPrior*sqrt(sum(array(wj)**2))
            wkrPrm[wkrId] = [float(w) for w in wj] + [float(tj)]
        return wkrPrm

    def sample_image_param(self, numImg, beta=.5, theta=.8):
        """
        Sample image parameters according to model.
        """
        dim = self.get_model_param()['dim']
        xis = {}
        for idx in range(numImg):
            xi = randn(dim)*theta
            xi += 1.0 if rand()<beta else -1.0
            xis[idx] = xi.tolist()
        return xis

    def sample_label(self, wkrPrm, imgPrm):
        dim = len(imgPrm)
        wj, tj = wkrPrm[:dim], wkrPrm[dim]
        xi = array(imgPrm)
        tj1, sj, wj1 = self.tw2tsw(tj, wj)
        return [bool(dot(xi+randn(dim)*sj,wj1) > tj1)]

    def tw2tsw(self, tj, wj):
        """
        Converts from the (tj, wj) convention to (tj, sj, wj).
        """
        return tw2tsw(tj, wj)
    
    def init_param_from_1d(self, numIter=30, noiseSig=1., noiseTrunc=2.,
                           xiTrunc = 2.):
        """
        Runs the 1d model on the data to initialize the 
        """
        dim = self.get_model_param()['dim']
        assert dim==2, "Only works when dimension is 2."
        m = Binary1dSignalModel(filename=self._dataFile)
        m.optimize_param()
        # set image parameters
        imgPrm = m.get_image_param(); numImg = len(imgPrm)
        newImgPrm = []
        pertVec = (array([-1, 1])/sqrt(2)).tolist()
        for id in range(numImg):
            xi = imgPrm[id][0]
            if xi>xiTrunc: xi=xiTrunc
            if xi<-xiTrunc: xi=-xiTrunc
            pert = randtn(-noiseTrunc, noiseTrunc)*noiseSig
            newImgPrm += [xi+pertVec[0]*pert, xi+pertVec[1]*pert]
        self.set_image_param(newImgPrm)
