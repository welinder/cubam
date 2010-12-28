from os.path import dirname, abspath, join, normpath
from ctypes import CDLL, c_char_p, c_void_p, c_double, c_int, cast, POINTER

# connect to the shared library (assumed to reside in ../cubamcpp.so)
libdir = normpath(join(dirname(abspath(__file__)), '..'))
annmodel = CDLL(abspath(join(libdir, 'cubamcpp.so')))

# set up function argument and return types
# this is needed to avoid 64/32 bit conversion errors
annmodel.setup_model.argtypes = [c_char_p]
annmodel.setup_model.restype = c_void_p

annmodel.clear_model.argtypes = [c_void_p]

annmodel.load_data.argtypes = [c_void_p, c_char_p]

annmodel.set_model_param.argtypes = [c_void_p, POINTER(c_double)]
annmodel.get_model_param.argtypes = [c_void_p, POINTER(c_double)]

annmodel.set_worker_param.argtypes = [c_void_p, POINTER(c_double)]
annmodel.set_image_param.argtypes = [c_void_p, POINTER(c_double)]
annmodel.get_worker_param.argtypes = [c_void_p, POINTER(c_double)]
annmodel.get_image_param.argtypes = [c_void_p, POINTER(c_double)]
 
annmodel.objective.argtypes = [c_void_p]
annmodel.objective.restype = c_double

annmodel.image_objective.argtypes = [c_void_p, c_int, POINTER(c_double),
                                     c_int, POINTER(c_double)]
annmodel.worker_objective.argtypes = [c_void_p, c_int, POINTER(c_double),
                                      c_int, POINTER(c_double)]
annmodel.gradient.argtypes = [c_void_p, POINTER(c_double)]

annmodel.get_num_wkr_lbls.argtypes = [c_void_p, POINTER(c_int)]
annmodel.get_num_img_lbls.argtypes = [c_void_p, POINTER(c_int)]

annmodel.get_num_wkrs.argtypes = [c_void_p]
annmodel.get_num_wkrs.restype = c_int
annmodel.get_num_imgs.argtypes = [c_void_p]
annmodel.get_num_imgs.restype = c_int
annmodel.get_num_lbls.argtypes = [c_void_p]
annmodel.get_num_lbls.restype = c_int

annmodel.get_model_param_len.argtypes = [c_void_p]
annmodel.get_model_param_len.restype = c_int
annmodel.get_worker_param_len.argtypes = [c_void_p]
annmodel.get_worker_param_len.restype = c_int
annmodel.get_image_param_len.argtypes = [c_void_p]
annmodel.get_image_param_len.restype = c_int
