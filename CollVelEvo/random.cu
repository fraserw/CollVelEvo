#include "cutil.h"
#include "cudpp.h"
#include <sys/time.h>



__global__ void convI2FK(unsigned int * in, float * out){
  int val = blockDim.x*blockIdx.x+threadIdx.x;

  out[val] = __int_as_float((in[val] & 0x7FFFFF)|0x3F800000)-1.0f;
}

void doRandGenFloat(float * randOut, unsigned int Size){
  struct timeval tim;
  gettimeofday(&tim, NULL);
  unsigned int seed = (unsigned int) tim.tv_sec + tim.tv_usec*100000.;

  unsigned int * outVI;
  CUDA_SAFE_CALL(cudaMalloc((void **) &outVI, Size*Size*sizeof(unsigned int)));

  CUDPPConfiguration config;
  config.op = CUDPP_ADD;
  config.datatype = CUDPP_UINT;
  config.algorithm = CUDPP_RAND_MD5;
  config.options = 0;

  CUDPPHandle randPlan = 0;
  CUDPPResult result;

  result = cudppPlan(&randPlan, config, Size*Size, 1, 0);

  cudppRandSeed(randPlan, seed);

  cudppRand(randPlan, outVI, Size*Size);

  convI2FK<<<Size,Size>>>(outVI,randOut);

  result = cudppDestroyPlan(randPlan);
  cudaFree(outVI);

}
