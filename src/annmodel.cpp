#include <cstring>

#include "utils.hpp"
#include "Binary1dSignalModel.hpp"
#include "BinaryNdSignalModel.hpp"
#include "annmodel.hpp"

using namespace std;

// TODO: handle errors better
// see http://www.parashift.com/c++-faq-lite/pointers-to-members.html
//#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))
//typedef void (Model::*ModelMemFn)(const char *filename);


EXPORTED MODEL_PTR setup_model(const char* model) {
  Model *ptr = 0;
  if(strcmp(model, "Binary1dSignalModel") == 0)
    ptr = new Binary1dSignalModel();
  if(strcmp(model, "BinaryNdSignalModel") == 0)
    ptr = new BinaryNdSignalModel();
  // TODO: insert try/catch block
  return (MODEL_PTR) ptr;
}

EXPORTED void clear_model(MODEL_PTR ptr) {
  if (ptr == 0)
    return;
  Model *mptr = (Model*) ptr;
  delete mptr; 
  mptr = 0; ptr = 0;
}

EXPORTED void load_data(MODEL_PTR ptr, const char *filename) {
  Model *mptr = (Model*) ptr;
  mptr->load_data(filename);
}

EXPORTED void set_model_param(MODEL_PTR ptr, double *prm) {
  Model *mptr = (Model*) ptr;
  mptr->set_model_param(prm);
}

EXPORTED void set_worker_param(MODEL_PTR ptr, double *prm) {
  Model *mptr = (Model*) ptr;
  mptr->set_worker_param(prm);
}

EXPORTED void set_image_param(MODEL_PTR ptr, double *prm) {
  Model *mptr = (Model*) ptr;
  mptr->set_image_param(prm);
}

EXPORTED void get_model_param(MODEL_PTR ptr, double *prm) {
  Model *mptr = (Model*) ptr;
  mptr->get_model_param(prm);
}

EXPORTED void get_worker_param(MODEL_PTR ptr, double *prm) {
  Model *mptr = (Model*) ptr;
  mptr->get_worker_param(prm);
}

EXPORTED void get_image_param(MODEL_PTR ptr, double *prm) {
  Model *mptr = (Model*) ptr;
  mptr->get_image_param(prm);
}

EXPORTED double objective(MODEL_PTR ptr) {
  Model *mptr = (Model*) ptr;
  return mptr->objective();
}

EXPORTED void image_objective(MODEL_PTR ptr, int imgId, double *prm, 
                              int nprm, double* obj) {
  Model *mptr = (Model*) ptr;
  mptr->image_objective(imgId, prm, nprm, obj);                                
}

EXPORTED void worker_objective(MODEL_PTR ptr, int wkrId, double *prm, 
                               int nprm, double* obj) {
  Model *mptr = (Model*) ptr;
  mptr->worker_objective(wkrId, prm, nprm, obj);                                
}

EXPORTED void gradient(MODEL_PTR ptr, double *grad) {
  Model *mptr = (Model*) ptr;
  mptr->gradient(grad);
}

EXPORTED void get_num_wkr_lbls(MODEL_PTR ptr, int *num) {
  Model *mptr = (Model*) ptr;
  mptr->get_num_wkr_lbls(num);
}

EXPORTED void get_num_img_lbls(MODEL_PTR ptr, int *num) {
  Model *mptr = (Model*) ptr;
  mptr->get_num_img_lbls(num);
}

EXPORTED int get_num_wkrs(MODEL_PTR ptr) {
  Model *mptr = (Model*) ptr;
  return mptr->get_num_wkrs();
}

EXPORTED int get_num_imgs(MODEL_PTR ptr) {
  Model *mptr = (Model*) ptr;
  return mptr->get_num_imgs();
}

EXPORTED int get_num_lbls(MODEL_PTR ptr) {
  Model *mptr = (Model*) ptr;
  return mptr->get_num_lbls();
}

EXPORTED int get_model_param_len(MODEL_PTR ptr) {
  Model *mptr = (Model*) ptr;
  return mptr->get_model_param_len();
}

EXPORTED int get_worker_param_len(MODEL_PTR ptr) {
  Model *mptr = (Model*) ptr;
  return mptr->get_worker_param_len();
}

EXPORTED int get_image_param_len(MODEL_PTR ptr) {
  Model *mptr = (Model*) ptr;
  return mptr->get_image_param_len();
}
