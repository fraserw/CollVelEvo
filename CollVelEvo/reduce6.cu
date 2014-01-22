#include<stdio.h>

//#include "cutil.h"

#define FADD(a,b) __fadd_rn(a,b)

#define sof sizeof(float)

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

///////////////////////////////////////////////////////////////////////
/**  parallel reduction Harris 07
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
    AQ: this routine should be called with 
       dim3 dimBlock(threads, 1, 1);
       dim3 dimGrid(numBlocks,1,1); but numBlocks = n/(2*threads)
   this routine meant to be run with numBlocks smaller than this by various factors of 2
       int smemSize=threads*sizeof(float);
        case 512:
            reduce6<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata,size); 
	break:
        case 256:
            reduce6<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata,size); 
*/
///////////////////////////////////////////////////////////////////////
/// g_idata input vector on device length n
/// g_odata output vector on device  length numBlocks
/// n length of vector
/// reduction kernel
template <unsigned int blockSize> // number of threads 
__global__ void
reduce6K(float* g_idata, float* g_odata, unsigned int n)
{
    extern volatile __shared__ float sdata[];
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;

    /// we reduce multiple elements per thread.  The number is determined by the 
    /// number of active thread blocks (via gridSize).  More blocks will result
    /// in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];  
        i += gridSize;
    } 
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; EMUSYNC; }
    }

    
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    __syncthreads();
}

/* input vector of length N, output vector of length n*/
void reduce6(float * dataV, float * odataV, int Size, int n){
  
  int N=Size*Size;

  cudaError_t err;

  int threads=Size/2;
  dim3 blockDim(threads); 
  dim3 gridDim(n);
  size_t sharedMemSize=threads*sof;

  switch(threads) { // may need more cases here 
  case  32:
    reduce6K<32> <<< gridDim, blockDim, sharedMemSize>>>(dataV, odataV, N);
    break;
  case  64:
    reduce6K<64> <<< gridDim, blockDim, sharedMemSize>>>(dataV, odataV, N);
    break;
  case 128:
    reduce6K<128> <<< gridDim, blockDim, sharedMemSize>>>(dataV, odataV, N);
    break;
  case 256:
    reduce6K<256> <<< gridDim, blockDim, sharedMemSize>>>(dataV, odataV, N);
    break;
  case 512:
    reduce6K<512> <<< gridDim, blockDim, sharedMemSize>>>(dataV, odataV, N);
    break;
  }
  err = cudaThreadSynchronize();
  if (err != cudaSuccess) printf("reduce6 kernel execution failed\n");
}
