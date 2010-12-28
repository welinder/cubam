#include "Model.hpp"
using namespace std;

Model::Model() {
  mNumWkrs = 0;
  mNumImgs = 0;
  mNumLbls = 0;
  mDataIsLoaded = false;
  mNumWkrLbls = 0;
  mNumImgLbls = 0;
}

Model::~Model() {
  this->clear_data();
}

void Model::clear_data() {
  delete [] mNumWkrLbls; mNumWkrLbls = 0;
  delete [] mNumImgLbls; mNumImgLbls = 0;
}

void Model::get_num_wkr_lbls(int *num) {
  for (int j=0; j<mNumWkrs; j++)
    num[j] = mNumWkrLbls[j];
}

void Model::get_num_img_lbls(int *num) {
  for (int i=0; i<mNumImgs; i++)
    num[i] = mNumImgLbls[i];
}
