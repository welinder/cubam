#ifndef __Model_hpp__
#define __Model_hpp__

class Model {
public:
  Model();
  virtual ~Model();

  virtual void set_model_param(double*) = 0;
  virtual void get_model_param(double*) = 0;

  virtual void set_worker_param(double*) = 0;
  virtual void set_image_param(double*) = 0;
  virtual void get_worker_param(double*) = 0;
  virtual void get_image_param(double*) = 0;
  virtual void reset_worker_param() = 0;
  virtual void reset_image_param() = 0;
  
  virtual void worker_objective(int, double*, int, double*) = 0;
  virtual void image_objective(int, double*, int, double*) = 0;

  virtual void load_data(const char *filename) = 0;
  virtual void clear_data() = 0;

  virtual double objective() = 0;
  virtual void gradient(double *grad) = 0;

  int get_num_wkrs() { return mNumWkrs; }
  int get_num_imgs() { return mNumImgs; }
  int get_num_lbls() { return mNumLbls; }
  
  virtual int get_worker_param_len() = 0;
  virtual int get_image_param_len() = 0;
  virtual int get_model_param_len() = 0;
  
  void get_num_wkr_lbls(int *num);
  void get_num_img_lbls(int *num);

protected:
  virtual void clear_worker_param() = 0;
  virtual void clear_image_param() = 0;

  int mNumWkrs;
  int mNumImgs;
  int mNumLbls;
  int *mNumWkrLbls;
  int *mNumImgLbls;
  bool mDataIsLoaded;
};

#endif
