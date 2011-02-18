from BinaryModel import *
from numpy import sign

class Binary1dSignalModel(BinaryModel):
  def __init__(self, filename=None, data=None):
    self._mdlPrmList = ['beta', 'sigx', 'sigw', 'muw', 'sigt']
    BinaryModel.__init__(self, filename, data)

  def image_objective_range(self, imgId, prm):
    plen = len(prm)
    pvec = (plen*c_double)(); pvec = cast(pvec, POINTER(c_double))
    for i in range(plen): pvec[i] = prm[i]
    vec = (plen*c_double)(); vec = cast(vec, POINTER(c_double))
    annmodel.image_objective(self.mPtr, c_int(imgId), pvec, c_int(plen), vec)
    return (vec[:plen], prm)
    
  def worker_objective_range(self, wkrId, prm=None, wj=None, tj=None):
    if prm is None:
      if tj is None or wj is None:
          tj = linspace(-1.5, 1.5, 30).tolist()
          wj = linspace(0.05, 3.0, 30).tolist()
      # create the gird
      wjs, tjs = meshgrid(wj, tj)
      wjs = concatenate(wjs).tolist()
      tjs = concatenate(tjs).tolist()
      prm = concatenate([wjs,tjs]).tolist()
    plen = len(prm)
    pvec = (plen*c_double)(); pvec = cast(pvec, POINTER(c_double))
    for i in range(plen): pvec[i] = prm[i]
    vlen = plen/2
    vec = (vlen*c_double)(); vec = cast(vec, POINTER(c_double))
    annmodel.worker_objective(self.mPtr, c_int(wkrId), pvec, c_int(plen), vec)
    res = vec[:vlen]
    if tj is None or wj is None:
      return (res, wjs, tjs)
    else:
      return (reshape(res, (len(wj), len(tj))), wjs, tjs)
    
  def get_worker_param(self, id=None):
    prm = self.get_worker_param_raw()
    nprm = len(prm)/2
    if id is None:
      return dict((id, [prm[id], prm[nprm+id]]) for id in range(nprm))
    else:
      return [prm[id], prm[nprm+id]]

  def get_image_param(self, id=None):
    prm = self.get_image_param_raw()
    if id is None:
      return dict((id, [prm[id]]) for id in range(len(prm)))
    else:
      return [prm[id]]
    
  def get_worker_var(self, wkrId):
    """
    Note: only works post-optimization
    """
    N = 50 # TODO: optimize this at some point
    bwj, btj = self.get_worker_param(wkrId)
    tj = linspace(btj-2, btj+2, N).tolist()
    wj = linspace(bwj+2, bwj-2, N).tolist()
    obj, wjs, tjs  = self.worker_objective_range(wkrId, wj=wj, tj=tj)
    obj = concatenate(obj)
    obj = exp(obj-max(obj))
    obj = array(obj)/sum(obj)
    dists = (array(wjs)-bwj)**2 + (array(tjs)-btj)**2
    return sqrt(sum(dists*obj))

  def get_image_var(self, imgId):
    N = 200 # TODO: optimize this at some point
    bxi = self.get_image_param(imgId)
    lb = min(-4, bxi-1.5); ub = max(4, bxi+1.5)
    xi = linspace(lb, ub, N).tolist()
    obj, xis = self.image_objective_range(imgId, xi)
    obj = obj-max(obj)
    obj = exp(obj)
    obj = array(obj)/sum(obj)
    dists = (array(xis)-bxi)**2
    return (sqrt(sum(dists*obj)), sum(obj[array(xis)>0]))

  def sample_worker_param(self, numWkr, tauPrior=0.8, advPrior=.01,
                          sigPrior=[1.5, .3], sigThresh=[.05, 3.],
                          tauThresh=[-2., 2.]):
    """
    Sample worker parameters according to model.
    """
    wkrPrm = {}
    for wkrId in range(numWkr):
      sj = gamma(sigPrior[0], sigPrior[1])
      # threshold the sj so we don't get too bad outliers
      sj = min([max([sj, sigThresh[0]]), sigThresh[1]])
      # sample angle uniformly in different intervals depending depending
      # on if the worker is averserial or not
      wj = (-1. if (rand() < advPrior) else 1.)/sj
      tj = randtn(*tauThresh)*tauPrior/sj
      wkrPrm[wkrId] = [float(wj), float(tj)]
    return wkrPrm

  def sample_image_param(self, numImg, beta=.5, theta=.5):
    """
    Sample image parameters according to model.
    """
    xis = randn(numImg)*theta
    ri = rand(numImg)<beta
    xis[ri==True] += 1
    xis[ri==False] -= 1
    return dict((idx, [xi]) for (idx, xi) in enumerate(xis))
  
  def sample_label(self, wkrPrm, imgPrm):
    wj, tj = wkrPrm
    xi = imgPrm[0]
    tj1, sj, wj1 = self.tw2tsw(tj, wj)
    return [bool( wj1*((xi+randn()*sj)-tj1) > 0. )]

  def tw2tsw(self, tj, wj):
    """
    Converts from the (tj, wj) convention to (tj, sj, wj).
    """
    return (tj/abs(wj), 1./abs(wj), sign(wj))

  def tsw2tw(self, tj, sj, wj):
    """
    Converts from the (tj, sj, wj) convention to (tj, wj).
    """
    return (tj/sj, 1./sj*wj)

  def get_labels(self):
    prm = self.get_image_param_raw()
    return dict((id, int(prm[id]>0.0)) for id in range(len(prm)))

