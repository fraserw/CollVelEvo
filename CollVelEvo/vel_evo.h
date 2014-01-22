__device__ float K(float m);
__device__ float E(float m);
__device__ float Gamma(float eccS, float incS);
__device__ void PvsK(float eccs, float incs, float * outArr);
__device__ void PVSHigh(float eccs, float incs, float gamma, float * outArr);
__device__ void IpvsK(float beta, float * outArr);
__global__ void deidtK(float * distV, float * massV, 
		       float * eccSquaredV, float * incSquaredV,
		       float * dedtij, float * didtij,
		       float sqrtao, float area, int threadInt);
__global__ void sumEccInc(float * dedtV, float * didtV,
			  float *eccSquaredV, float * incSquaredV,
			  float * velDispSV,
			  float dt, float ao, float Omega);
void getVel(float * distV, float * massV,
	    float * eccSquaredV, float * incSquaredV,
	    float * velDispSV, 
	    float area, float dt, float ao, float Omega,
	    int Size);
