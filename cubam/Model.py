import os
import yaml
from numpy import array, linspace, meshgrid, concatenate, reshape, exp, \
  sqrt, max, zeros, log, argmax, pi, tile, r_, ceil, floor, isnan, isinf
from numpy.random import rand, randn, gamma
from scipy.optimize import fmin_slsqp, fmin_l_bfgs_b
from ctypes import CDLL, c_char_p, c_void_p, c_double, c_int, cast, POINTER
from annmodel import annmodel
from utils import randtn, write_data_file

## main model class
class Model:
  """
  Abstract base class for models.
  """
  def __init__(self, filename=None, data=None):
    className = self.__class__.__name__
    self.mPtr = annmodel.setup_model(c_char_p(className))
    self.wkrIds = {}
    self.imgIds = {}
    if filename:
      self.load_data(filename)
    elif not data is None:
      filename = write_data_file(data)
      self.load_data(filename)
      os.remove(filename)
    
  def __del__(self):
    annmodel.clear_model(self.mPtr)
    
  def load_data(self, filename, skipyaml=False):
    yamlfile = "%s.yaml" % filename[:-4]
    if filename[-4:]=='.txt' and os.path.exists(yamlfile) and not skipyaml:
      prm = yaml.load(open(yamlfile))
      self.imgIds = prm['imgIds']
      self.wkrIds = prm['wkrIds']
    filename = c_char_p(filename)
    annmodel.load_data(self.mPtr, filename)
    
  def get_num_wkrs(self):
    return annmodel.get_num_wkrs(self.mPtr)
    
  def get_num_imgs(self):
    return annmodel.get_num_imgs(self.mPtr)
    
  def get_num_lbls(self):
    return annmodel.get_num_lbls(self.mPtr)
  
  def set_model_param(self, raw=[], prm=None):
    """
    Sets model parameters.

    Arguments:
      - `raw`: raw parameter vector
      - `prm`: hash of model parameter values to be changed
    """
    if not prm is None:
      # prm is set, ignore raw and set the
      oprm = self.get_model_param()
      for (key, val) in prm.iteritems(): oprm[key] = val
      raw = []
      for key in self._mdlPrmList: raw.append(oprm[key])
    # add the raw vector
    plen = annmodel.get_model_param_len(self.mPtr)
    assert len(raw) == plen, \
      "Raw parameter vector must be of length %d" % plen
    self._lib_set_vec('set_model_param', c_double, raw)
    
  def set_worker_param(self, raw):
    plen = annmodel.get_worker_param_len(self.mPtr)
    assert len(raw) == plen, \
      "Raw parameter vector must be of length %d" % plen
    self._lib_set_vec('set_worker_param', c_double, raw)
    
  def set_image_param(self, raw):
    plen = annmodel.get_image_param_len(self.mPtr)
    assert len(raw) == plen, \
      "Raw parameter vector must be of length %d" % plen
    self._lib_set_vec('set_image_param', c_double, raw)
    
  def get_model_param(self):
    plen = annmodel.get_model_param_len(self.mPtr)
    vec = self._lib_get_vec('get_model_param', c_double, plen)
    prm = {}
    for i in range(plen):
      key = self._mdlPrmList[i]
      prm[key] = vec[i]
    return prm
  
  def get_worker_param_raw(self):
    plen = annmodel.get_worker_param_len(self.mPtr)
    return self._lib_get_vec('get_worker_param', c_double, plen)
  
  def get_image_param_raw(self):
    plen = annmodel.get_image_param_len(self.mPtr)
    return self._lib_get_vec('get_image_param', c_double, plen)
    
  def get_worker_param(self, id=None):
    pass
  
  def get_image_param(self, id=None):
    pass
  
  # TODO: load and save parameters
  
  def optimize_worker_param(self):
    x0 = array(self.get_worker_param_raw())
    # res = fmin_slsqp(self.worker_objective, x0,
    #                  fprime=self.worker_gradient,
    #                  iprint=2, full_output=1, iter=10000)
    res = fmin_l_bfgs_b(self.worker_objective, x0, 
                        fprime=self.worker_gradient,
                        iprint=-1, maxfun=100)
    self.set_worker_param(res[0])
    return res
  
  def optimize_image_param(self):
    MAX_RESAMPLE_TRIES = 10
    for trial in range(MAX_RESAMPLE_TRIES):
        grad = self.image_gradient()
        if any(isnan(grad)) or any(isinf(grad)) or isinf(self.objective()):
            self.set_image_param((0.1*randn(len(grad))).tolist())
        else:
            break
    x0 = array(self.get_image_param_raw())
    # res = fmin_slsqp(self.image_objective, x0,
    #                  fprime=self.image_gradient,
    #                  iprint=2, full_output=1, iter=10000)
    res = fmin_l_bfgs_b(self.image_objective, x0, 
                        fprime=self.image_gradient,
                        iprint=-1, maxfun=100)
    self.set_image_param(res[0])
    return res
  
  def optimize_param(self, numIter=30, options=None, verbose=False):
    for n in range(numIter):
      if verbose: print "  - iteration %d/%d" % (n+1, numIter)
      self.optimize_image_param()
      self.optimize_worker_param()
  
  def objective(self, prm=None):
    n = annmodel.get_image_param_len(self.mPtr)
    if not prm is None:
      self.set_worker_param(prm[n:])
      self.set_image_param(prm[:n])
    return annmodel.objective(self.mPtr)
  
  def worker_objective(self, prm=None):
    if not prm is None: self.set_worker_param(prm)
    return annmodel.objective(self.mPtr)

  def image_objective(self, prm=None):
    if not prm is None: self.set_image_param(prm)
    return annmodel.objective(self.mPtr)
    
  def image_objective_range(self, imgId, prm):
    pass
  
  def worker_objective_range(self, wkrId, prm):
    pass
        
  def gradient(self, prm=None):
    n = annmodel.get_image_param_len(self.mPtr)
    if not prm is None:
      self.set_worker_param(prm[n:])
      self.set_image_param(prm[:n])
    glen = annmodel.get_worker_param_len(self.mPtr) + n
    return self._lib_get_vec('gradient', c_double, glen)
  
  def worker_gradient(self, prm=None):
    if not prm is None: self.set_worker_param(prm)
    n = annmodel.get_image_param_len(self.mPtr)
    grad = self.gradient()
    return array(grad[n:])

  def image_gradient(self, prm=None):
    if not prm is None: self.set_image_param(prm)
    n = annmodel.get_image_param_len(self.mPtr)
    grad = self.gradient()
    return array(grad[:n])
    
  def get_num_wkr_lbls(self):
    n = self.get_num_wkrs()
    return self._lib_get_vec('get_num_wkr_lbls', c_int, n)
    
  def get_num_img_lbls(self):
    n = self.get_num_imgs()
    return self._lib_get_vec('get_num_img_lbls', c_int, n)
    
  def _lib_get_vec(self, fname, vtype, vlen):
    vec = (vlen*vtype)()
    vec = cast(vec, POINTER(vtype))
    fn = getattr(annmodel, fname)
    fn(self.mPtr, vec)
    return vec[:vlen]

  def _lib_set_vec(self, fname, vtype, vec):
    vlen = len(vec)
    cvec = (vlen*vtype)()
    cvec = cast(cvec, POINTER(vtype))
    for i in range(vlen):
      cvec[i] = vec[i]
    fn = getattr(annmodel, fname)
    fn(self.mPtr, cvec)
