#include <stdexcept>
#include <fstream>
#include <cstdio>
#include <cstring>
#include "utils.hpp"

#include "BinaryModel.hpp"

using namespace std;

BinaryModel::BinaryModel() {
  mWkrLbls = 0;
  mImgLbls = 0;
  mLabels = 0;
}

void BinaryModel::clear_data() {
  if (!mDataIsLoaded)
    return;
  for (int j=0; j<mNumWkrs; j++)
    delete [] mWkrLbls[j];
  delete [] mWkrLbls; mWkrLbls = 0;
  for (int i=0; i<mNumImgs; i++)
    delete [] mImgLbls[i];
  delete [] mImgLbls; mImgLbls = 0;
  delete [] mLabels; mLabels = 0;
  mDataIsLoaded = false;
}

// TODO: error checking here so that we know when a file was corrupt
void BinaryModel::load_data(const char *filename) {
  if (mDataIsLoaded)
    throw runtime_error("You must clear the old data before loading new data.");
  // read data file
  ifstream inFile;
  inFile.open(filename, ios::in);
  if (!inFile)
    throw runtime_error("Unable to open data file.");
  // read the dataset size and set up labels
  char line[LINELEN+1];
  inFile.getline(line, LINELEN);
  sscanf(line, "%d %d %d\n", &mNumImgs, &mNumWkrs, &mNumLbls);
  mNumWkrLbls = new int[mNumWkrs];
  for (int j=0; j<mNumWkrs; j++) mNumWkrLbls[j] = 0;
  mNumImgLbls = new int[mNumImgs];
  for (int i=0; i<mNumImgs; i++) mNumImgLbls[i] = 0;
  mLabels = new int[3*mNumLbls]; // 3 since (i, j, label)
  // read the labels
  int i, j, label;
  int idx = 0;
  while(!inFile.eof()) {
    inFile.getline(line,100);
    if (strlen(line)<5) // need to fit at least 3 columns
      continue;
    sscanf(line, "%d %d %d\n", &i, &j, &label);
    mLabels[idx] = i; mLabels[idx+1] = j; mLabels[idx+2] = label;
    mNumWkrLbls[j] += 1;
    mNumImgLbls[i] += 1;
    idx += 3;
  }
  inFile.close();  
  // allocate mem for the image and worker labels
  mImgLbls = new int*[mNumImgs];
  for (i=0; i<mNumImgs; i++)
    mImgLbls[i] = new int[mNumImgLbls[i]*2];
  mWkrLbls = new int*[mNumWkrs];
  for (j=0; j<mNumWkrs; j++)
    mWkrLbls[j] = new int[mNumWkrLbls[j]*2];
  // replicate the labels for access from wkrs and imgs
  int *wkrIdx = new int[mNumWkrs];
  for (int j=0; j<mNumWkrs; j++) wkrIdx[j] = 0;
  int *imgIdx = new int[mNumImgs];
  for (int i=0; i<mNumImgs; i++) imgIdx[i] = 0;
  for (int k=0; k<mNumLbls; k++) {
    idx = k*3;
    i = mLabels[idx]; j = mLabels[idx+1]; label = mLabels[idx+2];
    // image label
    (mImgLbls[i])[2*imgIdx[i]] = j;
    (mImgLbls[i])[2*imgIdx[i]+1] = label;
    imgIdx[i] += 1;
    // worker label
    (mWkrLbls[j])[2*wkrIdx[j]] = i;
    (mWkrLbls[j])[2*wkrIdx[j]+1] = label;
    wkrIdx[j] += 1;
  }
  delete [] wkrIdx; delete [] imgIdx; // temporary vars
  // since we have no. of workers and images, we can reset params
  mDataIsLoaded = true; // this must come before reset to avoid exceptions
  reset_worker_param();
  reset_image_param();
}
