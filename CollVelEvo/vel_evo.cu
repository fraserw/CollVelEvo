#include <stdio.h>
//#include "cutil.h"
#include "vel_evo.h"
#include "reduce.h"

#define piOver2 1.5707963267948966192313216916398
#define pi 3.1415926535897932384626433832795028841971693993
#define THREESOLARMASS 5.96676e33
#define SM2N56 1.7830036719332545e-28 //SOLARMASS^(-5/6)
#define G 6.674e-8
#define sqrtG 0.000258340860105

#define FMUL(a,b) __fmul_rn(a,b)
#define FADD(a,b) __fadd_rn(a,b)



/* wf - Ohtsuki and Stewart 2002 velocity evolution prescription */

/*tenth order approximation of the first complete elliptical integral.
  This is accurate to better (usually much) than 1% for m<0.9, and to 1% for 
  0.9<m<1.0*/
__device__ float K(float m) {
  float out = 1.0;
  float m2=m*m;
    float x = m2;
  out = FADD(out,FMUL(0.25,x));
  x*=m2;
  out = FADD(out,FMUL(0.14062f,x));
  x*=m2;
  out = FADD(out,FMUL(0.09765625f,x));
  x*=m2;
  out = FADD(out,FMUL(0.0747680664062f,x));
  x*=m2;
  out = FADD(out,FMUL(0.0605621337891f,x));
  x*=m2;
  out = FADD(out,FMUL(0.0508890151978f,x));
  x*=m2;
  out = FADD(out,FMUL(0.0438787937164f,x));
  x*=m2;
  out = FADD(out,FMUL(0.0385653460398f,x));
  x*=m2;
  out = FADD(out,FMUL(0.0343993364368f,x));
  return piOver2*out;

}

/*tenth order approximation of the second complete elliptical integral
  This is accurate to better (usually much) than 1% for m<0.9, and to 1% for
  0.9<m<1.0*/
__device__ float E(float m) {
  float out = 1.0;
  float m2=m*m;
  float x = m2;
  out = FADD(-FMUL(0.25,x),out);
  x*=m2;
  out = FADD(-FMUL(0.046875f,x),out);
  x*=m2;
  out = FADD(-FMUL(0.01953125f,x),out);
  x*=m2;
  out = FADD(-FMUL(0.0106811523438f,x),out);
  x*=m2;
  out = FADD(-FMUL(0.00672912597656f,x),out);
  x*=m2;
  out = FADD(-FMUL(0.00462627410889f,x),out);
  x*=m2;
  out = FADD(-FMUL(0.00337529182434f,x),out);
  x*=m2;
  out = FADD(-FMUL(0.00257102306932f,x),out);
  x*=m2;
  out = FADD(-FMUL(0.00202349037863f,x),out);
  x*=m2;
  out = FADD(-FMUL(0.00163396848075f,x),out);
  return piOver2*out;
}



__device__ void PVSHigh(float eccs, float incs, float gamma, float * outArr) {
  float e = sqrtf(eccs);
  float i =sqrtf(incs);
  float beta = i/e;
  float gamma2=gamma*gamma;

  float I[4];

  IpvsK(beta, I);

  float pei=1.0f/(pi*e*i);
  float aPVS =  72.0f*I[0]*pei;
  float aQVS =  72.0f*I[1]*pei;
  float aPDF = 576.0f*I[2]*pei;
  float aQDF = 576.0f*I[3]*pei;
  

  float multi=0.0;
  if (not isinf(gamma2)) {
    multi = log1pf(gamma2);
  } else {
    multi = 2.0f*logf(gamma);
  }
  outArr[0] = aPVS*multi;
  outArr[1] = aQVS*multi;
  outArr[2] = aPDF*multi;
  outArr[3] = aQDF*multi;
  
}


/* wf calculate IPVS,IQVS,IPDF,IQDF (eqn 21) from Ohtsuki et al. 2002.
   Use 9th order Gauss Hermite Quadrature.
*/
__device__ void IpvsK(float beta, float * outArr) {
  float lambda1s=powf(0.01591988025f,2.0f);
  float lambda2s=__powf(0.08198444635f,2.0f);
  float lambda3s=__powf(0.1933142865f,2.0f);
  float lambda4s=__powf(0.3378732883f,2.0f);
  float lambda5s=__powf(0.5f,2.0f);
  float lambda6s=__powf(0.6621267117f,2.0f);
  float lambda7s=__powf(0.8066857135f,2.0f);
  float lambda8s=__powf(0.91801555365f,2.0f);
  float lambda9s=__powf(0.98408011975f,2.0f);
  float w1=0.08127438836f;
  float w2=0.1806481606f;
  float w3=0.2606106964f;
  float w4=0.3123470770f;
  float w5=0.3302393550f;

  float x=sqrtf(3.0*(1.0-lambda1s))/2.;
  float K1=K(x);
  float E1=E(x);
  x=sqrtf(3.0*(1.0-lambda2s))/2.;
  float K2=K(x);
  float E2=E(x);
  x=sqrtf(3.0*(1-lambda3s))/2.;
  float K3=K(x);
  float E3=E(x);
  x=sqrtf(3.0*(1-lambda4s))/2.;
  float K4=K(x);
  float E4=E(x);
  x=sqrtf(3.0*(1-lambda5s))/2.;
  float K5=K(x);
  float E5=E(x);
  x=sqrtf(3.0*(1-lambda6s))/2.;
  float K6=K(x);
  float E6=E(x);
  x=sqrtf(3.0*(1-lambda7s))/2.;
  float K7=K(x);
  float E7=E(x);
  x=sqrtf(3.0*(1-lambda8s))/2.;
  float K8=K(x);
  float E8=E(x);
  x=sqrtf(3.0*(1-lambda9s))/2.;
  float K9=K(x);
  float E9=E(x);


  float denom1 = 1.0f/(beta + (1.0/beta - beta)* lambda1s);
  float denom2 = 1.0f/(beta + (1.0/beta - beta)* lambda2s);
  float denom3 = 1.0f/(beta + (1.0/beta - beta)* lambda3s);
  float denom4 = 1.0f/(beta + (1.0/beta - beta)* lambda4s);
  float denom5 = 1.0f/(beta + (1.0/beta - beta)* lambda5s);
  float denom6 = 1.0f/(beta + (1.0/beta - beta)* lambda6s);
  float denom7 = 1.0f/(beta + (1.0/beta - beta)* lambda7s);
  float denom8 = 1.0f/(beta + (1.0/beta - beta)* lambda8s);
  float denom9 = 1.0f/(beta + (1.0/beta - beta)* lambda9s);

  float p3l1 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda1s)));
  float p3l2 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda2s)));
  float p3l3 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda3s)));
  float p3l4 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda4s)));
  float p3l5 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda5s)));
  float p3l6 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda6s)));
  float p3l7 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda7s)));
  float p3l8 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda8s)));
  float p3l9 = 1.0f/(FADD(1.0f,FMUL(3.0f,lambda9s)));


  float out;
  out = w1*(5.0f*K1-12.0f*(1.0f-lambda1s)*E1*p3l1)*denom1;
  out+= w2*(5.0f*K2-12.0f*(1.0f-lambda2s)*E2*p3l2)*denom2;
  out+= w3*(5.0f*K3-12.0f*(1.0f-lambda3s)*E3*p3l3)*denom3;
  out+= w4*(5.0f*K4-12.0f*(1.0f-lambda4s)*E4*p3l4)*denom4;
  out+= w5*(5.0f*K5-12.0f*(1.0f-lambda5s)*E5*p3l5)*denom5;
  out+= w4*(5.0f*K6-12.0f*(1.0f-lambda6s)*E6*p3l6)*denom6;
  out+= w3*(5.0f*K7-12.0f*(1.0f-lambda7s)*E7*p3l7)*denom7;
  out+= w2*(5.0f*K8-12.0f*(1.0f-lambda8s)*E8*p3l8)*denom8;
  out+= w1*(5.0f*K9-12.0f*(1.0f-lambda9s)*E9*p3l9)*denom9;
  out*=0.5;

  outArr[0] = out;
  
  out =  w1*((K1 - 12.0*lambda1s*E1*p3l1)*denom1);
  out += w2*((K2 - 12.0*lambda2s*E2*p3l2)*denom2);
  out += w3*((K3 - 12.0*lambda3s*E3*p3l3)*denom3);
  out += w4*((K4 - 12.0*lambda4s*E4*p3l4)*denom4);
  out += w5*((K5 - 12.0*lambda5s*E5*p3l5)*denom5);
  out += w4*((K6 - 12.0*lambda6s*E6*p3l6)*denom6);
  out += w3*((K7 - 12.0*lambda7s*E7*p3l7)*denom7);
  out += w2*((K8 - 12.0*lambda8s*E8*p3l8)*denom8);
  out += w1*((K9 - 12.0*lambda9s*E9*p3l9)*denom9);
  out *= 0.5;

  outArr[1] = out;
  
  out =  w1*((E1*(1.0f-lambda1s)*p3l1)*denom1);
  out += w2*((E2*(1.0f-lambda2s)*p3l2)*denom2);
  out += w3*((E3*(1.0f-lambda3s)*p3l3)*denom3);
  out += w4*((E4*(1.0f-lambda4s)*p3l4)*denom4);
  out += w5*((E5*(1.0f-lambda5s)*p3l5)*denom5);
  out += w4*((E6*(1.0f-lambda6s)*p3l6)*denom6);
  out += w3*((E7*(1.0f-lambda7s)*p3l7)*denom7);
  out += w2*((E8*(1.0f-lambda8s)*p3l8)*denom8);
  out += w1*((E9*(1.0f-lambda9s)*p3l9)*denom9);
  out *= 0.5;

  outArr[2] = out;

  out =  w1*(E1*lambda1s*p3l1)*denom1;
  out += w2*(E2*lambda2s*p3l2)*denom2;
  out += w3*(E3*lambda3s*p3l3)*denom3;
  out += w4*(E4*lambda4s*p3l4)*denom4;
  out += w5*(E5*lambda5s*p3l5)*denom5;
  out += w4*(E6*lambda6s*p3l6)*denom6;
  out += w3*(E7*lambda7s*p3l7)*denom7;
  out += w2*(E8*lambda8s*p3l8)*denom8;
  out += w1*(E9*lambda9s*p3l9)*denom9;
  out *= 0.5;

  outArr[3] = out;
  
}



__device__ float Gamma(float eccS, float incS){
  return 0.083333333333333333333333333f*(eccS+incS)*sqrtf(incS);
}

/*the second term was borked but then fixed by adding in the infinity check 
  if statements
*/
__device__ void PvsK(float eccs, float incs, float * outArr) {

  float gamma = Gamma(eccs,incs);
  float gamma2ten = 10.0f*gamma*gamma;

  float C1=0.0;
  float C2=0.0;
  float C3=0.0;

  float cost = gamma2ten/eccs;
  if (not isinf(cost)) C1 = log1pf(cost)/cost;
  cost = gamma2ten*sqrtf(eccs);
  if (not isinf(cost)) C2 = log1pf(cost)/cost;
  if (not isinf(gamma2ten)) C3 = log1pf(gamma2ten)/gamma2ten;

  float getArrLow[4];
  float getArrHigh[4]={0.0f,0.0f,0.0f,0.0f};
  PVSHigh(eccs,incs,gamma,getArrHigh);


  float sfi=sqrtf(incs);
  getArrLow[0] = 73.0f;
  getArrLow[1] = FMUL(4.0f,sfi)+FMUL(FMUL(0.2f,powf(eccs,1.5f)),sfi);
  getArrLow[2] = 10.0f*eccs;
  getArrLow[3] = 10.0f*incs;
  if (isinf(getArrLow[1])) getArrLow[1]=1.e30; //put here to fake large numbers. Could do better with logs when calculating getArrLow[1] and outArr[1]. But for now, this is a satisfactory cluge to make sure that ourArr[1] comes sufficiently close to 0!


  outArr[0]=FADD(FMUL(C1,getArrLow[0]),getArrHigh[0]);
  outArr[1]=FADD(FMUL(C2,getArrLow[1]),getArrHigh[1]);
  outArr[2]=FADD(FMUL(C3,getArrLow[2]),getArrHigh[2]);
  outArr[3]=FADD(FMUL(C3,getArrLow[3]),getArrHigh[3]);


}


/* this function calculates the change in the mean eccentricity
and inclination
of particle i by particle j, where i and j are defined by the
thread and block numbers
*/
__global__ void deidtK(float * distV, float * massV, 
		       float * eccSquaredV, float * incSquaredV,
		       float * dedtij, float * didtij,
		       float sqrtao, float area, int threadInt){
  
  int i = blockIdx.x;
  int j = threadIdx.x;
  int Size = blockDim.x;
  j+= threadInt*Size;
  int val = i*gridDim.x +j;

  float arr[4]={0.0f,0.0f,0.0f,0.0f};

  float mi=massV[i];
  float mj=massV[j];
  float mij=FADD(mi,mj); 
  float hij2 = powf(mij/(THREESOLARMASS),0.6666666666666667f);

  float ei2 = eccSquaredV[i];
  float ej2 = eccSquaredV[j];
  float eccs = ei2+ej2;

  float ii2 = incSquaredV[i];
  float ij2 = incSquaredV[j];
  float incs = ii2+ij2;

  float eccstilde= eccs/hij2;
  float incstilde= incs/hij2;
  PvsK(eccstilde, incstilde, arr);


  
  float Nsj = distV[j*3+1]/area;   //surface density
  float pmij= __powf(mij,-0.666666666666667f);
  

  float valToMakeFaster = Nsj*mj*pmij;

  didtij[val] = (mj*arr[1] + (mj*ij2 - mi*ii2)*arr[3]/incs)*valToMakeFaster;
 
  dedtij[val] = (mj*arr[0]+(mj*ej2 - mi*ei2)*arr[2]/eccs)*valToMakeFaster;
  
  
  
}



/* add the de and di values to the average value arrays. 
   Call as <<<1,dimbloack>>>*/
__global__ void sumEccInc(float * dedtV, float * didtV,
			  float *eccSquaredV, float * incSquaredV,
			  float * velDispSV,
			  float dt, float ao, float Omega){
  int j = threadIdx.x;
  float c=ao*Omega;
  float sqrtao=sqrtf(ao);
  float c2=c*c;
  double constant=0.23112042478354491f*sqrtG*sqrtao*SM2N56;

  float e=FMUL((double) dedtV[j],((double) dt)*constant) + eccSquaredV[j];  
  eccSquaredV[j]=e;

  float i=FMUL((double) didtV[j],((double) dt)*constant) + incSquaredV[j];
  incSquaredV[j]=i;

  velDispSV[2*j] =   0.625*e*c2; // (5/8), c2 is (ao*omega)^2
  velDispSV[2*j+1] = 0.5*i*c2;

  
}


__global__ void CallIPVSK(float beta, float * out){
  IpvsK(beta,out);
}


//masses can't be less than ~1.e-4 
//number in a bin can't be higher than ~1.e29
void getVel(float * distV, float * massV,
	    float * eccSquaredV, float * incSquaredV,
	    float * velDispSV, 
	    float area, float dt, float ao, float Omega,
	    int Size){


  float * didtij;
  float * dedtij;
  float * didtV;
  float * dedtV;

  cudaMalloc((void **) &didtV, Size*sizeof(float));
  cudaMalloc((void **) &dedtV, Size*sizeof(float));
  cudaMalloc((void **) &didtij, Size*Size*sizeof(float));
  cudaMalloc((void **) &dedtij, Size*Size*sizeof(float));

  //cudaMemset(didtV,0,Size*sizeof(float));
  //cudaMemset(dedtV,0,Size*sizeof(float));

  int nThread = min(128,Size);
  dim3 gridDim(Size);
  dim3 blockDimSmall(nThread);
  int numThreadGroups = Size/nThread;
  for (int j=0;j<numThreadGroups;j++) {
    deidtK<<<gridDim,blockDimSmall>>>(distV, massV, eccSquaredV,incSquaredV,
				      dedtij, didtij,
				      sqrtf(ao),area, j);
  }


  /*
  float crap1[Size*Size];
  float crap2[Size*Size];
  cudaMemcpy(crap2,didtij,Size*Size*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(crap1,dedtij,Size*Size*sizeof(float),cudaMemcpyDeviceToHost);
  
  for (int i=0;i<10;i++){
    //printf("%d %.10e %.10e \n",i,crap1[i],crap2[i]);
    printf("%d %.10e \n",i,crap2[i]);
    }
  exit(1);
  */


  //sum up all the eccentricity/inc changes
  reduce3(didtij,didtV,Size,Size);
  reduce3(dedtij,dedtV,Size,Size);

  /*
  float scrap[Size];
  cudaMemcpy(scrap,didtV,Size*sizeof(float),cudaMemcpyDeviceToHost);
  for (int i=0;i<Size;i++){
    printf(" %d %.10e \n",i,scrap[i]);
  }
  */

  sumEccInc<<<1,Size>>>(dedtV,didtV, eccSquaredV, incSquaredV, 
			velDispSV, dt, ao, Omega);

  cudaFree(didtV);
  cudaFree(dedtV);
  cudaFree(didtij);
  cudaFree(dedtij);
  
 
}
