#ifndef __UTILS_HPP__
#define __UTILS_HPP__

// no of chars in line buffer
#define LINELEN 100

// mathematical helpers
#define PI 3.14159265
#define LOGNORM(xx,mu,sig) (-0.5*(log(2.0*PI)+2.0*log(sig)   \
                                  +(xx-mu)*(xx-mu)/sig/sig))
#define NORMAL(xx,sig) exp(-0.5*(xx)/(sig)/(sig)) \
                       /sqrt(2.0*PI)*(sig)
double cdf(double x);


#endif
