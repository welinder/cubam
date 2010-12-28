#include <stdexcept>
#include <cmath>
#include "utils.hpp"

#include "BinaryNdSignalModel.hpp"

using namespace std;

void BinaryNdSignalModel::set_model_param(double *prm) {
  mBeta = prm[0];
  mSigX = prm[1];
  mSigW = prm[2];
  mMuW = prm[3];
  mSigT = prm[4];
  if (mDataIsLoaded && (prm[5] != mDim))
    throw runtime_error("Cannot set dimension when data is loaded.");
  mDim = int(prm[5]);
}

void BinaryNdSignalModel::get_model_param(double *prm) {
  prm[0] = mBeta;
  prm[1] = mSigX;
  prm[2] = mSigW;
  prm[3] = mMuW;
  prm[4] = mSigT;
  prm[5] = double(mDim);
}

void BinaryNdSignalModel::reset_worker_param() {
  if (!mDataIsLoaded)
    throw runtime_error("You must load data before resetting parameters.");
  clear_worker_param();
  int nElements = mNumWkrs*mDim;
  mWjs = new double[nElements];
  for (int j=0; j<nElements; j++)
    mWjs[j] = 1.0;
  mTjs = new double[mNumWkrs];
  for (int j=0; j<mNumWkrs; j++)
    mTjs[j] = 0.0;
}

void BinaryNdSignalModel::reset_image_param() {
  if (!mDataIsLoaded)
    throw runtime_error("You must load data before resetting parameters.");
  clear_image_param();
  int nElements = mNumImgs*mDim;
  mXis = new double[nElements];
  for (int i=0; i<nElements; i++)
    mXis[i] = 0.0;
}

void BinaryNdSignalModel::set_image_param(double *xis) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumImgs*mDim;
  for (int i=0; i<nElements; i++)
    mXis[i] = xis[i];
}

void BinaryNdSignalModel::set_worker_param(double *vars) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumWkrs*mDim;
  for (int j=0; j<nElements; j++)
    mWjs[j] = vars[j];
  for (int j=0; j<mNumWkrs; j++)
    mTjs[j] = vars[j+nElements];
}

void BinaryNdSignalModel::get_image_param(double *xis) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumImgs*mDim;
  for (int i=0; i<nElements; i++)
    xis[i] = mXis[i];
}

void BinaryNdSignalModel::get_worker_param(double *vars) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumWkrs*mDim;
  for (int j=0; j<nElements; j++)
    vars[j] = mWjs[j];
  for (int j=0; j<mNumWkrs; j++)
    vars[j+nElements] = mTjs[j];
}

double BinaryNdSignalModel::objective() {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  double obj = 0.0;
  // compute the xi prior
  for(int i=0; i<mNumImgs; i++) {
    double x0sq = 0.0;
    double x1sq = 0.0;
    for(int d=0; d<mDim; d++) {
      x0sq += (mXis[i*mDim+d]+1.0)*(mXis[i*mDim+d]+1.0);
      x1sq += (mXis[i*mDim+d]-1.0)*(mXis[i*mDim+d]-1.0);
    }
    obj += (-0.5*double(mDim)*log(2.0*PI*mSigX*mSigX)
            + log(mBeta*exp(-0.5*x1sq/(mSigX*mSigX))
                  + (1.-mBeta)*exp(-0.5*x0sq/(mSigX*mSigX))));
  }
  // compute the wj prior
  for(int j=0; j<mNumWkrs; j++)
    for(int d=0; d<mDim; d++)
      obj += LOGNORM(mWjs[j*mDim+d], mMuW, mSigW);
  // compute the tj prior
  for(int j=0; j<mNumWkrs; j++)
    obj += LOGNORM(mTjs[j], 0.0, mSigT);
  // compute the shared terms
  int idx, i, j, lij;
  for(int k=0; k<mNumLbls; k++) {
    idx = k*3;
    i = mLabels[idx];
    j = mLabels[idx+1];
    lij = mLabels[idx+2];
    double cdfarg = 0.0;
    for(int d=0; d<mDim; d++)
      cdfarg += mXis[i*mDim+d]*mWjs[j*mDim+d];
    cdfarg -= mTjs[j];
    if(lij == 0) {
      if(cdfarg<0.0)
        obj += log(1.0-cdf(cdfarg));
      else
        obj += log(cdf(-cdfarg));
    } else {
      if(cdfarg<0.0)
        obj += log(cdf(cdfarg));
      else
        obj += log(1.0-cdf(-cdfarg));
    }
  }
  return -obj;
}

void BinaryNdSignalModel::worker_objective(int wkrId, double *prm, 
                                           int nprm, double* obj) {
  // wkrId is the id of the worker we're interested in
  // prm is a list of [wjs, tjs] we're computing the objective for
  // nprm is the length of prm, and obj is the resulting objective fn
  // list (of length nprm/2)
  
  // compute xi prior sum (shared by all terms)
  double xiprior = 0.0;
  for(int i=0; i<mNumWkrLbls[wkrId]; i++) {
    int idx = (mWkrLbls[wkrId])[2*i]; // the image idx
    double x0sq = 0.0;
    double x1sq = 0.0;
    for(int d=0; d<mDim; d++) {
      x0sq += (mXis[idx*mDim+d]+1.0)*(mXis[idx*mDim+d]+1.0);
      x1sq += (mXis[idx*mDim+d]-1.0)*(mXis[idx*mDim+d]-1.0);
    }
    xiprior += (-0.5*double(mDim)*log(2.0*PI*mSigX*mSigX)
            + log(mBeta*exp(-0.5*x1sq/(mSigX*mSigX))
                  + (1.-mBeta)*exp(-0.5*x0sq/(mSigX*mSigX))));
  }
  // add wkr prm specific prior
  int npts = nprm/(1+mDim); // since we have (wj, tj) pairs in the prm list
  int toffset = npts*mDim;
  for(int j=0; j<npts; j++) {
    obj[j] = xiprior + LOGNORM(prm[toffset+j], 0.0, mSigT);
    for(int d=0; d<mDim; d++)
      obj[j] += LOGNORM(prm[j*mDim+d], mMuW, mSigW);
  }
    
  // compute the shared terms
  int endidx = mNumWkrLbls[wkrId]*2;
  for(int idx=0; idx<endidx; idx+=2) {
    int i = (mWkrLbls[wkrId])[idx];
    int lij = (mWkrLbls[wkrId])[idx+1];
    for(int j=0; j<npts; j++) {
      double cdfarg = 0.0;
      for(int d=0; d<mDim; d++)
        cdfarg += mXis[i*mDim+d]*prm[j*mDim+d];
      cdfarg -= prm[toffset+j];
      if(lij == 0) {
        if(cdfarg<0.0)
          obj[j] += log(1.0-cdf(cdfarg));
        else
          obj[j] += log(cdf(-cdfarg));
      } else {
        if(cdfarg<0.0)
          obj[j] += log(cdf(cdfarg));
        else
          obj[j] += log(1.0-cdf(-cdfarg));
      }
    }
  }
}

void BinaryNdSignalModel::image_objective(int imgId, double *prm, 
                                          int nprm, double* obj) {
  // imgId is the id of the image we're interested in
  // prm is a list of xi's we're computing the objective for
  // nprm is the length of prm, and obj is the resulting objective fn
  // list (also of length nprm)
  
  // add worker priors (shared by all terms)
  double wkrprior = 0.0;
  for(int j=0; j<mNumImgLbls[imgId]; j++) {
    int idx = (mImgLbls[imgId])[2*j];
    wkrprior += LOGNORM(mTjs[idx], 0.0, mSigT);
    for(int d=0; d<mDim; d++)
      wkrprior += LOGNORM(mWjs[idx*mDim+d], mMuW, mSigW);
  }
  // compute image related priors based on the prm vector
  int npts = nprm/mDim;
  for(int i=0; i<npts; i++) {
    double x0sq = 0.0;
    double x1sq = 0.0;
    for(int d=0; d<mDim; d++) {
      x0sq += (prm[i*mDim+d]+1.0)*(prm[i*mDim+d]+1.0);
      x1sq += (prm[i*mDim+d]-1.0)*(prm[i*mDim+d]-1.0);
    }
    obj[i] = wkrprior + (-0.5*double(mDim)*log(2.0*PI*mSigX*mSigX)
                + log(mBeta*exp(-0.5*x1sq/(mSigX*mSigX))
                + (1.-mBeta)*exp(-0.5*x0sq/(mSigX*mSigX))));
  }
  // compute the shared terms
  int endidx = mNumImgLbls[imgId]*2;
  for(int idx=0; idx<endidx; idx+=2) {
    int j = (mImgLbls[imgId])[idx];
    int lij = (mImgLbls[imgId])[idx+1];
    for(int i=0; i<nprm; i++) {
      double cdfarg = 0.0;
      for(int d=0; d<mDim; d++)
        cdfarg += prm[i*mDim+d]*mWjs[j*mDim+d];
      cdfarg -= mTjs[j];
      if(lij == 0) {
        if(cdfarg<0.0)
          obj[i] += log(1.0-cdf(cdfarg));
        else
          obj[i] += log(cdf(-cdfarg));
      } else {
        if(cdfarg<0.0)
          obj[i] += log(cdf(cdfarg));
        else
          obj[i] += log(1.0-cdf(-cdfarg));
      }
    }
  }
}

void BinaryNdSignalModel::gradient(double *grad) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  // assumes that *grad is a list of length mNumImgs+2*mNumWkrs
  // the order is assumed to be [xis, wjs, tjs]
  int gradLen = mDim*mNumImgs + (1+mDim)*mNumWkrs;
  for(int i=0; i<gradLen; i++)
    grad[i] = 0.0;
  // compute the xi prior gradient
  for(int i=0; i<mNumImgs; i++) {
    double x0sq = 0.0;
    double x1sq = 0.0;
    for(int d=0; d<mDim; d++) {
      x0sq += (mXis[i*mDim+d]+1.0)*(mXis[i*mDim+d]+1.0);
      x1sq += (mXis[i*mDim+d]-1.0)*(mXis[i*mDim+d]-1.0);
    }
    x0sq = NORMAL(x0sq, mSigX);
    x1sq = NORMAL(x1sq, mSigX);
    for(int d=0; d<mDim; d++)
      grad[i*mDim+d] = (mBeta*(mXis[i*mDim+d]-1.0)*x1sq + 
        (1.0-mBeta)*(mXis[i*mDim+d]+1.0)*x0sq)
        /mSigX/mSigX/ (mBeta*x1sq +(1.0-mBeta)*x0sq);
  }
  // compute the wj & tj prior gradients
  int woffset = mNumImgs*mDim;
  int toffset = woffset + mNumWkrs*mDim;
  for(int j=0; j<mNumWkrs; j++) {
    grad[toffset+j] = mTjs[j]/mSigT/mSigT;
    for(int d=0; d<mDim; d++)
      grad[woffset+j*mDim+d] = (mWjs[j*mDim+d]-mMuW)/mSigW/mSigW;
  }
  // compute the shared terms
  int idx, i, j, lij;
  for(int k=0; k<mNumLbls; k++) {
    idx = k*3;
    i = mLabels[idx];
    j = mLabels[idx+1];
    lij = mLabels[idx+2];
    double cdfarg = 0.0;
    for(int d=0; d<mDim; d++)
      cdfarg += mXis[i*mDim+d]*mWjs[j*mDim+d];
    cdfarg -= mTjs[j];
    double lambda_ij;
    if(lij == 0) {
      if(cdfarg<0.0)
        lambda_ij = -1.0/(1.0-cdf(cdfarg));
      else
        lambda_ij = -1.0/cdf(-cdfarg);
    } else {
      if(cdfarg<0.0)
        lambda_ij = 1.0/cdf(cdfarg);
      else
        lambda_ij = 1.0/(1.0-cdf(-cdfarg));
    }
    double philambda_ij = exp(-0.5*cdfarg*cdfarg)/sqrt(2.0*PI)*lambda_ij;
    // add shared components to gradients
    for(int d=0; d<mDim; d++) {
      grad[i*mDim+d] -= mWjs[j*mDim+d]*philambda_ij;
      grad[woffset+j*mDim+d] -= mXis[i*mDim+d]*philambda_ij;
    }
    grad[toffset+j] += philambda_ij;
  }
}
