#ifndef __Binary1dSignalModel_hpp__
#define __Binary1dSignalModel_hpp__

#include "BinarySignalModel.hpp"

class Binary1dSignalModel : public BinarySignalModel {
public:
  void set_worker_param(double *vars);
  void set_image_param(double *xis);
  void get_worker_param(double *vars);
  void get_image_param(double *xis);
  void reset_worker_param();
  void reset_image_param();
  
  void worker_objective(int wkrId, double *prm, int nprm, double* obj);
  void image_objective(int imgId, double *prm, int nprm, double* obj);
  
  virtual int get_worker_param_len() { return mNumWkrs*2; }
  virtual int get_image_param_len() { return mNumImgs; }
  virtual int get_model_param_len() { return 5; }
  
  double objective();
  void gradient(double *grad);
};

#endif
