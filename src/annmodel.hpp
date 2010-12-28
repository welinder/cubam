#ifndef __ANNMODEL_HPP__
#define __ANNMODEL_HPP__

#define EXPORTED extern "C"
typedef void* MODEL_PTR;

EXPORTED MODEL_PTR setup_model(const char*);
EXPORTED void clear_model(MODEL_PTR ptr);

EXPORTED void load_data(MODEL_PTR ptr, const char* filename);

EXPORTED void set_model_param(MODEL_PTR ptr, double *prm);
EXPORTED void get_model_param(MODEL_PTR ptr, double *prm);

EXPORTED void set_worker_param(MODEL_PTR ptr, double *prm);
EXPORTED void set_image_param(MODEL_PTR ptr, double *prm);
EXPORTED void get_worker_param(MODEL_PTR ptr, double *prm);
EXPORTED void get_image_param(MODEL_PTR ptr, double *prm);

EXPORTED double objective(MODEL_PTR ptr);
EXPORTED void image_objective(MODEL_PTR ptr, int imgId, double *prm, 
                              int nprm, double* obj);
EXPORTED void worker_objective(MODEL_PTR ptr, int wkrId, double *prm, 
                               int nprm, double* obj);
EXPORTED void gradient(MODEL_PTR ptr, double *grad);

EXPORTED void get_num_wkr_lbls(MODEL_PTR ptr, int *num);
EXPORTED void get_num_img_lbls(MODEL_PTR ptr, int *num);

EXPORTED int get_num_wkrs(MODEL_PTR ptr);
EXPORTED int get_num_imgs(MODEL_PTR ptr);
EXPORTED int get_num_lbls(MODEL_PTR ptr);

EXPORTED int get_model_param_len(MODEL_PTR ptr);
EXPORTED int get_worker_param_len(MODEL_PTR ptr);
EXPORTED int get_image_param_len(MODEL_PTR ptr);

#endif
