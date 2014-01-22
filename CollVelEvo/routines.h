#define pi 3.1415926535897932384626433832795028841971693993

#define FADD(a,b) __fadd_rn(a,b)
#define FMUL(a,b) __fmul_rn(a,b)
#define FDIV(a,b) __fdiv_rn(a,b)




float bigFunc(float * distV,
	      float * rirjNjV, 
	      float * SrirjNjV,
	      float * massV,
	      float * QeV,
	      float * QdsV,
	      float * GammaV,
	      float * VescV,
	      float * ranValsV,
	      float * fMescV,
	      float * numAddV,
	      float QeMin,
	      float rmin,
	      float fke,
	      float fshat,
	      float alpha,
	      float qc,
	      float maxCratRat,
	      float rho,
	      float tsMax,
	      int Size);

__global__ void numCollKernel(float * distV,
			      float * rirjNjV, 
			      float * SrirjNjV,
			      float * ranValsV,
			      float ts,
			      float * numCollV,
			      float * numAddV,
			      int Size);

__global__ void outcomeKernel(float * distV,
			      float * rirjNjV, 
			      float * SrirjNjV,
			      float * massV,
			      float * QeV,
			      float * QdsV,
			      float * GammaV,
			      float * VescV,
			      float * ranValsV,
			      float * fMescV,
			      float * numCollV,
			      float * numAddV,
			      float * numAddFromCrateringV,
			      int * craterTargetBinsV,
			      float * diffMassV,
			      float ts,
			      float QeMin,
			      float rmin,
			      float fke,
			      float fshat,
			      float alpha,
			      float qc,
			      float maxCratRat,
			      float rho,
			      int Size);


__global__ void numAddReduce(float * numAddV,
			     float * numAddReduceV,
			     int Size);

__global__ void rsiK(float * distK, float * VescV, float * velDispSV, float * rsiV, int Size);
__global__ void rirjNjK(float * rsiV, float * distV, float * velDispSV, 
			float * rirjNjV, int Size,float LG);
__global__ void getQeGK(float * velDispSV,
			float * distV,
			float * massV,
			float * QdsV,
			float * VescV,
			float rho,
			float Qo,
			float B,
			float a,
			float bb,
			float Gmin,
			float * QeV,
			float * GV,
			int Size);

__global__ void getVescFK(float * massV,
			  float * distV,
			  float Grav,
			  float Vmin,
			  float kcrat,
			  float * VescV,
			  float * fMescV,
			  int Size);
__global__ void getMassQdsK(float * distV, 
			    float rho, 
			    float Qo, 
			    float B, 
			    float a, 
			    float bb, 
			    float * massV, 
			    float * QdsV);
__global__ void getMassJumpK(float * massV,
			     float * MassJumpV,
			     int Size);
__global__ void ulTriangle(float * ranVals, int Size);
