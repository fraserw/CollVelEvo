#include "routines.h"
#include <stdio.h>
#include <cutil.h>
#include "reduce.h"




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
	      int Size) {

  int kk=0.;

  float ts=1.0e6;

  float * diffMassV;
  CUDA_SAFE_CALL(cudaMalloc((void **) &diffMassV, Size*Size*sizeof(float)));

  float * numAddFromCrateringV;
  CUDA_SAFE_CALL(cudaMalloc((void **) &numAddFromCrateringV, Size*Size*sizeof(float)));

  int * craterTargetBinsV;
  CUDA_SAFE_CALL(cudaMalloc((void **) &craterTargetBinsV, Size*Size*sizeof(int)));

  float * numAddRedV;
  CUDA_SAFE_CALL(cudaMalloc((void **) &numAddRedV,Size*Size*sizeof(float)));

  float * numAddTotV;
  CUDA_SAFE_CALL(cudaMalloc((void **) & numAddTotV, Size*sizeof(float)));

  float * numCollV;
  CUDA_SAFE_CALL(cudaMalloc((void **) & numCollV, Size*Size*sizeof(float)));
  
  CUDA_SAFE_CALL(cudaMemset((void *) numAddV, 0, Size*Size*Size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemset((void *) diffMassV, 0, Size*Size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemset((void *) numAddFromCrateringV, 0, Size*Size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemset((void *) craterTargetBinsV, 0, Size*Size*sizeof(int)));
  CUDA_SAFE_CALL(cudaMemset((void *) numAddTotV, 0, Size*sizeof(int)));


  //determine the number of collisions, store in ncoll
  //also store in numAdd the number of objects removed by collision
  numCollKernel<<<Size,Size>>>(distV,rirjNjV, SrirjNjV,ranValsV,ts,numCollV,numAddV, Size) ;


  outcomeKernel<<<Size,Size>>>(distV,
			       rirjNjV, 
			       SrirjNjV,
			       massV,
			       QeV,
			       QdsV,
			       GammaV,
			       VescV,
			       ranValsV,
			       fMescV,
			       numCollV,
			       numAddV,
			       numAddFromCrateringV,
			       craterTargetBinsV,
			       diffMassV,
			       ts,
			       QeMin,
			       rmin,
			       fke,
			       fshat,
			       alpha,
			       qc,
			       maxCratRat,
			       rho,
			       Size);




  
  float numAddFromCratering[Size*Size];
  int craterTargetBins[Size*Size];
  float diffMass[Size*Size];
  float numAddTot[Size];
  float dist[Size*3];
  float numColl[Size*Size];
  float numAdd[Size*Size*Size];
  

  //Sum up all the numbers added from catastrophic collisions
  //  as well as those ejecta fragments from cratering
  numAddReduce<<<8,Size/8>>>(numAddV,numAddTotV,Size);

  //now on the CPU side add in the components due to cratering
  cudaMemcpy(numAddTot,numAddTotV,Size*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(numAddFromCratering,numAddFromCrateringV,Size*Size*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(craterTargetBins,craterTargetBinsV,Size*Size*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(diffMass,diffMassV,Size*Size*sizeof(float),cudaMemcpyDeviceToHost);

  for (int i=0;i<Size;i++){
    for (int j=0;j<Size;j++){
      if (diffMass[i*Size+j]!=0){
	printf("%d %d %g\n",i,j,diffMass[i*Size+j]);
      }
    }
  }
  exit(1);
  for (int i=0;i<Size;i++){
    for (int j=0;j<Size;j++){
      if (numAddFromCratering[i*Size+j]!=0){
	printf("%d %d %g %d\n",i,j,numAddFromCratering[i*Size+j],craterTargetBins[i*Size+j]);
      }
    }
  }
  exit(1);

  


  for (int i=0;i<Size;i++){
    for (int j=0;j<Size;j++){
      kk=i*Size+j;
      numAddTot[craterTargetBins[kk]]+=numAddFromCratering[kk];
    }
  }
  


  ////
  /////
  //////
  //above this, we are done, below, not so much
  //////
  /////
  ////

  //on the cpu side, add to numAdd all the objects added to other bins 
  //  via cratering
  
  cudaMemcpy(numAdd,numAddV,Size*Size*Size*sizeof(float),cudaMemcpyDeviceToHost);


  
  cudaMemcpy(diffMass,diffMassV,Size*Size*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(dist,distV,3*Size*sizeof(float),cudaMemcpyDeviceToHost);

  for (int i=0;i<Size;i++){
    printf("%d %g %g %g  \n",i,dist[3*i+1],numAddTot[i],numAddTot[i]/dist[3*i+1]);
  }
  exit(1);





  
  
  cudaFree(numAddTotV);
  cudaFree(diffMassV);
  cudaFree(numAddFromCrateringV);
  cudaFree(craterTargetBinsV);

  exit(1);
  float timeStep=0.;
  return timeStep;
}


__global__ void numCollKernel(float * distV,
			      float * rirjNjV, 
			      float * SrirjNjV,
			      float * ranValsV,
			      float ts,
			      float * numCollV,
			      float * numAddV,
			      int Size) {

     int i = blockIdx.x;
     int j = threadIdx.x;
     int Size2=Size*Size;

     int val=i*Size+j;
     float x,y,ncoll;

     if (i<j) {
       numCollV[val]=0.0;
     } else {
       x=rirjNjV[val];
       y=rirjNjV[j*Size+i];

       //ncoll=ts*distV[3*i+1]*x * (1.0 + ts*(y-SrirjNjV[j]));
       ncoll=FMUL(ts,FMUL(distV[3*i+1],x));
       ncoll*=(1.0 + FMUL(ts,(y-SrirjNjV[j])));

       if (ncoll<ranValsV[val]) {
	 ncoll=0.;
       }
       else if (ncoll<1.e8) {
	 x=trunc(ncoll);
	 if (x>ranValsV[val]) {
	   ncoll=x+1;
	 } else {
	   ncoll=x;
	 }
       }

       numAddV[val+j*Size2]-=ncoll;
       numAddV[val+i*Size2]-=ncoll;
       numCollV[val]=ncoll;
     }
}



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
			      int Size) {
  

     int i = blockIdx.x;
     int j = threadIdx.x;

     int val=i*Size+j;

     float n;
     float ncoll;
     float rlrf;
     float GammaNCorr;
     float qd;
     float rt;
     float front,back;
     float mcrat;
     int Size2=Size*Size;
     float fragMass,massOfRemainingBody,diffMass;
     int targetBin;
     float ke;

     if (i>=j) {
       ke=QeV[val]-(QeMin*QdsV[i]);

       ncoll = numCollV[val];

       if (ke > 0) { //in the catastrophic disruption regime
	 fragMass=ncoll*(massV[i]+massV[j]);
	 
	 rt=powf( FADD(powf(distV[3*i],(float) 3.0), powf(distV[3*j],(float) 3.0)) ,(float) 1./3.);
	 if (rt<distV[3*i]){
	   rt=distV[3*i];
	 }

	 rlrf=powf(GammaV[val],(float) 1./3.)*rt;

	 if (rlrf<=rmin) {
	   n=FMUL(ncoll,FDIV(FADD(massV[i],massV[j]),massV[0]));
	   if (n<1.e8) {
	     n=round(n);
	   }
	   numAddV[val]+=n;
	   fragMass-=n*massV[0];

	 } else {

	   qd=(4.+GammaV[val])/(1.+GammaV[val]);

	   GammaNCorr=1./(1.0 - powf(rmin/rlrf,(float) 4.0-qd));


	   for (int k=0;k<i+1;k++){
	     front=min(rlrf,distV[3*k]+distV[3*k+2]/2.);
	     back =min(rlrf,distV[3*k]-distV[3*k+2]/2.);
	     
	     n=FMUL(ncoll,FMUL(GammaNCorr,FMUL(powf(rlrf,(float) qd-1.0),(powf(back,(float) 1.0-qd)-powf(front, (float) 1.0-qd)))));
	     if (n<1.e8){
	       n=round(n);
	     }
	       
	     numAddV[val+k*Size2]+=n;
	   
	     fragMass-=n*massV[k];
	   }
	 }

	 diffMassV[val]=n/ncoll;//fragMass/massV[i];///ncoll/massV[i];

       } /*else {  //cratering type collision

	 //determine basic cratering paramaters
	 massOfRemainingBody=massV[i]+massV[j];

	 mcrat=fminf(maxCratRat*(massV[i]+massV[j]),QeV[val]*alpha*fshat);

	 rlrf=powf((float) (4.-qc)*mcrat*3./((qc-1.)*4*pi*rho),(float) 1./3.);
	 
	 //distribute the fragments
	 if (rlrf<=rmin) {
	   n=ncoll*mcrat*fMescV[val]/massV[0];
	   if (n<1.e8) {
	     n=trunc(n);
	   }
	   numAddV[val]=+n;
	   massOfRemainingBody-=n*massV[0]/ncoll;

	 } else {

	   GammaNCorr=1./(1.0 - powf(rmin/rlrf,(float) 4-qc));


	   for (int k=0;k<i+1;k++) {
	     front=min(rlrf,distV[3*k]+distV[3*k+2]/2.);
	     back =min(rlrf,distV[3*k]-distV[3*k+2]/2.);
	     
	     n=ncoll*fMescV[val]*GammaNCorr*powf(rlrf,(float) qc-1)*(powf(back,(float) 1.0-qc) - powf(front, (float) 1.0-qc));
	     
	     if (n<1.e8) {
	       n=trunc(n);
	     }
	     numAddV[val+k*Size2]+=n;
	     
	     massOfRemainingBody-=n*massV[k]/ncoll;
	   }
	 }

	   
	 for (int k=1;k<Size;k++) {
	   targetBin=k;
	   if (massOfRemainingBody/massV[targetBin] <1) {
	     targetBin-=1;
	     break;
	   }
	 }
	 
	 n=ncoll*massOfRemainingBody/massV[targetBin];
	 if (n<1.e8) {
	   n=trunc(n);
	 }
	 numAddFromCrateringV[val]=n;
	 craterTargetBinsV[val]=targetBin;
	 
	 diffMass=n*(massOfRemainingBody-massV[targetBin]);
	 diffMassV[val]=diffMass;
	 }*/
     }
     
}

__global__ void numAddReduce(float * numAddV,
			     float * numAddTotV,
			     int Size) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  int n = blockDim.x;
  int k=i*n+j;
  int Size2=Size*Size;

  float num=0.;
  for (int I=0;I<Size;I++){
    for (int J=0;J<Size;J++){
      num+=numAddV[I*Size+J+k*Size2];
    }
  }
  
  numAddTotV[k]=num;
}



__global__ void rsiK(float * distK, float * VescV, float * velDispSV, float * rsiV, int Size){
     
     int i = blockIdx.x;
     int j = threadIdx.x;
     
     int val=i*Size+j;

     float Ve=pow(VescV[val],2);
     float VS=velDispSV[2*i]+velDispSV[2*i+1] + velDispSV[2*j]+velDispSV[2*j+1];

     rsiV[val] =( powf(distK[i*3],(float) 2.)+powf(distK[j*3],(float) 2.))*(1+Ve/VS);
     //__syncthreads();
}



__global__ void rirjNjK(float * rsiV, float * distV, float * velDispSV, 
			float * rirjNjV, int Size,float LG){
			
     
     int i = blockIdx.x;
     int j = threadIdx.x;

     int val=i*Size+j;
     float c;

     float logvel = 0.5f*__log10f(velDispSV[2*i]+velDispSV[2*i+1] + velDispSV[2*j]+velDispSV[2*j+1]);
     //for the real simulations
     if (i!=j) {
       c=powf((float) 10.0,(__log10f(rsiV[val])+__log10f(distV[j*3+1]) + logvel + LG));
     } else {
       c=0.5*powf((float) 10.0,(__log10f(rsiV[val])+__log10f( fmaxf(distV[j*3+1]-1,0.0)  ) + logvel + LG));
     }


     //for A=constant
     //if (i!=j) {
     //	rirjNjV[val]=distV[j*3+1];
     //} else {
     // rirjNjV[val]=0.5*fmaxf(distV[j*3+1]-1,0.0);}

     //for A=i+j
     //if (i!=j) {
     //	rirjNjV[val]=(powf(distV[i*3],3.0)+powf(distV[j*3],3.0))*distV[j*3+1];
     //} else {
     //   rirjNjV[val]=0.5*(powf(distV[i*3],3.0)+powf(distV[j*3],3.0))*fmaxf(distV[j*3+1]-1,0.0);}

     //for A=i*j
     //if (i!=j) {
     //	rirjNjV[val]=(powf(distV[i*3],3.0)*powf(distV[j*3],3.0))*distV[j*3+1];
     //} else {
     //   rirjNjV[val]=0.5*(powf(distV[i*3],3.0)*powf(distV[j*3],3.0))*fmaxf(distV[j*3+1]-1,0.0);}
     rirjNjV[val]=c;
     //__syncthreads();
}


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
			int Size){

  int i = blockIdx.x;
  int j = threadIdx.x;
  int val = i*Size+j;

  float ve,vCollS,qqq,gamma,mvi,mvj;

  ve=VescV[val];
  vCollS=fmaxf(ve*ve, velDispSV[2*i]+velDispSV[2*i+1]+velDispSV[2*j]+velDispSV[2*j+1]);
  
  mvi=massV[i];
  mvj=massV[j];
    
  //center of mass energy per unit mass is 0.25*mi*mj*v**2/(mi+mj)**2
  QeV[val]=FMUL(FMUL(0.25 , FMUL(FDIV(mvi, FADD(mvi,mvj) ), mvj) ), vCollS);
    
  qqq=QeV[val]/mvi;
  gamma = (1.0 -(10./18.)*(qqq/QdsV[i]));
  
  
  if (gamma<Gmin){
    GV[val]=Gmin;
  } else {
    GV[val]=gamma;
  }
}


__global__ void getVescFK(float * massV,
			  float * distV,
			  float Grav,
			  float Vmin,
			  float kcrat,
			  float * VescV,
			  float * fMescV,
			  int Size){
  int i = blockIdx.x;
  int j = threadIdx.x;

  int val = i*Size+j;
  float vesq;
  
  vesq = powf(2*Grav*(massV[i]+massV[j])/(distV[i*3]+distV[j*3]), 0.5);
  VescV[val]  = vesq;

  if (vesq>=Vmin){
    fMescV[val]=powf(vesq/Vmin,-kcrat);
  } else {
    fMescV[val]=1.0;
  }
}


__global__ void getMassQdsK(float * distV, 
			    float rho, 
			    float Qo, 
			    float B, 
			    float a, 
			    float bb, 
			    float * massV, 
			    float * QdsV){
  int i = threadIdx.x;

  float r = distV[i*3];

  float qds = Qo*powf(r,a)+B*rho*powf(r,bb);

  float mass = (4./3.)*pi*rho*powf(r,3.0);

  massV[i]=mass;
  QdsV[i]=qds;
}


__global__ void getMassJumpK(float * massV,
			     float * MassJumpV,
			     int Size){
  int i = threadIdx.x;
  
  if (i==0){
    MassJumpV[0]=-massV[0];
    MassJumpV[1]=massV[1]-massV[0];
  } else if (i==(Size-1)) {
    MassJumpV[2*i]=-massV[i]+massV[i-1];
    MassJumpV[2*i+1]=massV[i];
  } else {
    MassJumpV[2*i]=massV[i-1]-massV[i];
    MassJumpV[2*i+1]=massV[i+1]-massV[i];
  }
}


__global__ void ulTriangle(float * ranVals, int Size){
   	   int i = blockIdx.x;
	   int j = threadIdx.x;
	   
	   if (j<i){
	       ranVals[i*Size+j]=ranVals[j*Size+i];
           }
}
