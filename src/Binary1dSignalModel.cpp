#include <stdexcept>
#include <cmath>
#include "utils.hpp"

#include "Binary1dSignalModel.hpp"

using namespace std;

void Binary1dSignalModel::reset_worker_param() {
  if (!mDataIsLoaded)
    throw runtime_error("You must load data before resetting parameters.");
  clear_worker_param();
  mWjs = new double[mNumWkrs];
  mTjs = new double[mNumWkrs];
  for (int j=0; j<mNumWkrs; j++) {
    mWjs[j] = 1.0;
    mTjs[j] = 0.0;
  }
}

void Binary1dSignalModel::reset_image_param() {
  if (!mDataIsLoaded)
    throw runtime_error("You must load data before resetting parameters.");
  clear_image_param();
  mXis = new double[mNumImgs];
  for (int i=0; i<mNumImgs; i++)
    mXis[i] = 1.0;
}

void Binary1dSignalModel::set_image_param(double *xis) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  for (int i=0; i<mNumImgs; i++)
    mXis[i] = xis[i];
}

void Binary1dSignalModel::set_worker_param(double *vars) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  for (int j=0; j<mNumWkrs; j++) {
    mWjs[j] = vars[j];
    mTjs[j] = vars[j+mNumWkrs];
  }
}

void Binary1dSignalModel::get_image_param(double *xis) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  for (int i=0; i<mNumImgs; i++)
    xis[i] = mXis[i];
}

void Binary1dSignalModel::get_worker_param(double *vars) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  for (int j=0; j<mNumWkrs; j++) {
    vars[j] = mWjs[j];
    vars[j+mNumWkrs] = mTjs[j];
  }
}

double Binary1dSignalModel::objective() {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  double obj = 0.0;
  // compute the xi prior
  for(int i=0; i<mNumImgs; i++) {
    double x0sq = (mXis[i]+1.0)*(mXis[i]+1.0);
    double x1sq = (mXis[i]-1.0)*(mXis[i]-1.0);
    obj += (-0.5*log(2.0*PI*mSigX*mSigX)
            + log(mBeta*exp(-0.5*x1sq/(mSigX*mSigX))
                  + (1.-mBeta)*exp(-0.5*x0sq/(mSigX*mSigX))));
  }
  // compute the wj prior
  for(int j=0; j<mNumWkrs; j++)
    obj += LOGNORM(mWjs[j], mMuW, mSigW);
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
    double cdfarg = mXis[i]*mWjs[j] - mTjs[j];
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

void Binary1dSignalModel::worker_objective(int wkrId, double *prm, 
                                           int nprm, double* obj) {
  // wkrId is the id of the worker we're interested in
  // prm is a list of [wjs, tjs] we're computing the objective for
  // nprm is the length of prm, and obj is the resulting objective fn
  // list (of length nprm/2)
  
  // compute xi prior sum (shared by all terms)
  double xiprior = 0.0;
  for(int i=0; i<mNumWkrLbls[wkrId]; i++) {
    int idx = (mWkrLbls[wkrId])[2*i]; // the image idx
    double x0sq = (mXis[idx]+1.0)*(mXis[idx]+1.0);
    double x1sq = (mXis[idx]-1.0)*(mXis[idx]-1.0);
    xiprior += (-0.5*log(2.0*PI*mSigX*mSigX)
                + log(mBeta*exp(-0.5*x1sq/(mSigX*mSigX))
                      + (1.-mBeta)*exp(-0.5*x0sq/(mSigX*mSigX))));
  }
  // add wkr prm specific prior
  int npts = nprm/2; // since we have (wj, tj) pairs in the prm list
  int toffset = npts;
  for(int j=0; j<npts; j++)
    obj[j] = xiprior + LOGNORM(prm[j], mMuW, mSigW)
      + LOGNORM(prm[toffset+j], 0.0, mSigT);
  // compute the shared terms
  int endidx = mNumWkrLbls[wkrId]*2;
  for(int idx=0; idx<endidx; idx+=2) {
    int i = (mWkrLbls[wkrId])[idx];
    int lij = (mWkrLbls[wkrId])[idx+1];
    for(int j=0; j<npts; j++) {
      double cdfarg = mXis[i]*prm[j] - prm[toffset+j];
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

void Binary1dSignalModel::image_objective(int imgId, double *prm, 
                                          int nprm, double* obj) {
  // imgId is the id of the image we're interested in
  // prm is a list of xi's we're computing the objective for
  // nprm is the length of prm, and obj is the resulting objective fn
  // list (also of length nprm)
  
  // add worker priors (shared by all terms)
  double wkrprior = 0.0;
  for(int j=0; j<mNumImgLbls[imgId]; j++) {
    int idx = (mImgLbls[imgId])[2*j];
    wkrprior += LOGNORM(mWjs[idx], mMuW, mSigW) 
                + LOGNORM(mTjs[idx], 0.0, mSigT);
  }
  // compute image related priors based on the prm vector
  for(int i=0; i<nprm; i++) {
    double x0sq = (prm[i]+1.0)*(prm[i]+1.0);
    double x1sq = (prm[i]-1.0)*(prm[i]-1.0);
    obj[i] = wkrprior + (-0.5*log(2.0*PI*mSigX*mSigX)
                + log(mBeta*exp(-0.5*x1sq/(mSigX*mSigX))
                + (1.-mBeta)*exp(-0.5*x0sq/(mSigX*mSigX))));
  }
  // compute the shared terms
  int endidx = mNumImgLbls[imgId]*2;
  for(int idx=0; idx<endidx; idx+=2) {
    int j = (mImgLbls[imgId])[idx];
    int lij = (mImgLbls[imgId])[idx+1];
    for(int i=0; i<nprm; i++) {
      double cdfarg = prm[i]*mWjs[j] - mTjs[j];
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

void Binary1dSignalModel::gradient(double *grad) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  // assumes that *grad is a list of length mNumImgs+2*mNumWkrs
  // the order is assumed to be [xis, wjs, tjs]
  int gradLen = mNumImgs + 2*mNumWkrs;
  for(int i=0; i<gradLen; i++)
    grad[i] = 0.0;
  // compute the xi prior gradient
  for(int i=0; i<mNumImgs; i++) {
    double x0sq = (mXis[i]+1.0)*(mXis[i]+1.0);
    double x1sq = (mXis[i]-1.0)*(mXis[i]-1.0);
    x0sq = NORMAL(x0sq, mSigX);
    x1sq = NORMAL(x1sq, mSigX);
    grad[i] = (mBeta*(mXis[i]-1.0)*x1sq + (1.0-mBeta)*(mXis[i]+1.0)*x0sq)
      /mSigX/mSigX/ (mBeta*x1sq +(1.0-mBeta)*x0sq);
  }
  // compute the wj & tj prior gradients
  int woffset = mNumImgs;
  int toffset = woffset + mNumWkrs;
  for(int j=0; j<mNumWkrs; j++) {
    grad[toffset+j] = mTjs[j]/mSigT/mSigT;
    grad[woffset+j] = (mWjs[j]-mMuW)/mSigW/mSigW;
  }
  // compute the shared terms
  int idx, i, j, lij;
  for(int k=0; k<mNumLbls; k++) {
    idx = k*3;
    i = mLabels[idx];
    j = mLabels[idx+1];
    lij = mLabels[idx+2];
    double cdfarg = mXis[i]*mWjs[j] - mTjs[j];
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
    grad[i] -= mWjs[j]*philambda_ij;
    grad[woffset+j] -= mXis[i]*philambda_ij;
    grad[toffset+j] += philambda_ij;
  }
}
