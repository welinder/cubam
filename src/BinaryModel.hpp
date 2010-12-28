#ifndef __BinaryModel_hpp_
#define __BinaryModel_hpp_

#include "Model.hpp"

class BinaryModel : public Model {
public:
  BinaryModel();
  
  void load_data(const char *filename);
  void clear_data();
  
protected:  
  int **mWkrLbls;
  int **mImgLbls;
  int *mLabels;
};

#endif
