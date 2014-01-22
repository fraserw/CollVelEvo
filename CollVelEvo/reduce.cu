#include "cutil.h"

#define FADD(a,b) __fadd_rn(a,b)

#define sof sizeof(float)

/* the array reduction kernel. See next below.
   */
__global__ void reduce3K(float * g_idata, float * g_odata) {
  extern __shared__ float sdata[];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  // do reduction in shared mem  //this is modified to version 3
  //>>+ is a bitwise divide by 2 below, it only takes 1 away from odd #'s
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {  
      if (tid < s) {
      	 //sdata[tid] += sdata[tid + s];
      	 sdata[tid]=FADD(sdata[tid],sdata[tid + s]);
      }	
      __syncthreads();
  }
  //write the final result back to global memory
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



/* the sum reductions function
   N is the number of elements to sum together
   n is the number of resultant elements
   data is the input data with dimension N*n
   odata is the output data with dimension n

   NOTE:N must be of size 2^i. ie. 2,4,8,16 etc.
*/
void reduce3(float * dataV, float * odataV, int N, int n){

     //sum up each of the N blocks
     // so block 0 sums up elements 0 through N-1 and 
     //places the result in element 0, 
     // block 1 sums up elements N through 2N-1 etc.
     // NS is the shared memory allocation size
     dim3 blockDim(N); 
     dim3 gridDim(n);
     
     size_t NS=N*sof;

     reduce3K<<<gridDim,blockDim,NS>>>(dataV,odataV);
}

