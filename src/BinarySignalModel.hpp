#ifndef __BinarySignalModel_hpp__
#define __BinarySignalModel_hpp__

#include "BinaryModel.hpp"

class BinarySignalModel : public BinaryModel {
public:
  BinarySignalModel();
  
  void set_model_param(double *prm);
  void get_model_param(double *prm);
  
  void clear_data();
  
protected:
  void clear_worker_param();
  void clear_image_param();

  double *mXis;
  double *mWjs;
  double *mTjs;
  double mBeta;
  double mSigX;
  double mSigW;
  double mMuW;
  double mSigT;
};

#endif
