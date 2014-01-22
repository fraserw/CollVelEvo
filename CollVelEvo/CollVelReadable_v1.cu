#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "routines.h"
#include "reduce.h"
//#include "reduce6.h"
//#include "tileroutines.h"
#include "random.h"
//#include "vel_evo.h"
#include "read_write.h"

#include "cutil.h"

#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)
#define MUL(a, b) __umul24(a, b)

#define arrSize 128    //if you get a crash from this remember "ulimit -s unlimited" bash command first
//#define bmin 0.5
//#define bratio 1.1  //1.14 is max, 1.13 is max from Ohtsuki
#define pi 3.1415926535897932384626433832795028841971693993
#define multiBin 0

#define sof sizeof(float)

typedef long long int INT64;

const int Nran=arrSize*arrSize;

/// physical constants
#define solarMass 1.98892e33
#define Grav 6.67e-8
#define AU 1.49568e13





void reset2(float * tnc){
     int i,j;
     for(i=0;i<arrSize;i++){
	for(j=0;j<arrSize;j++){
	    tnc[i*arrSize+j]=0.0;
	}
     }
}
void reset1(float * tnc){
     int i;
     for(i=0;i<arrSize;i++){
     	tnc[i]=0.0;
     }
}



//this returns the Qemin value given the cratering/disruption radius percentage
float Qrat(float Rrat){
      return 18./10.*(1.-pow(Rrat, (float) 3));
}
     
int main(int argc, char ** argv){
    //used for output
    char outFileStr[100];

    //for CUDA itself
    int useDev = 0;

    double runTime=0.0;
    int outCount=0;
    int whichi=0;

    //for the random numbers

    //ranVals is used to randomize the N-collision rounding 
    //        -only needs to be on the video card
    //ranVals2 is used to randomize the bin reductions
    //        -only needs to be on the HOST
    //ranVals3 is used to move objects up and down from cratering
    //        -only needed on the videocard
    //float * ranVals;
    float * ranVals2;
    //float * ranVals3;
    float *ranValsV;
    float * ranValsV2;
    float * ranValsV3;

    //end of random numbers stuff


    float * distV;
    float * rsiV;
    float * rirjNjV;
    float * rirjNjVCC; //this one is rirjNj(i,j)*CatColl(i,j)
    float * trueNumColV;
    float * trueNumColIncreasedLargestV; //tracks collisions moving an object up in mass that aren't disruptive    
    float * SrirjNjV;
    float * totColV;
    float * QdsV;
    float * QV;
    float * GV;
    float * LRFV;
    float * CatCollV; //this one tracks the number of catastrophic disruptions
    float * CatCollDisV; //tracks the number of added material from CatCollV
    float * CratCollV; //tracks the number of non-disruptive collisions
    float * CratCollIncreasedLargestObjectV; //tracks collisions moving an object up in mass that aren't disruptive
    float * sNumAddedV;
    float * sNumCatV;
    float * VescV;
    float * fMescV;
    float * LRFcratV;
    float * McratV;
    float * AccMassV;
    float * MassJumpV;
    float * MassDiffCratV;
    float * massV;
    float * QeV;
    float * numAddV;

    float dist[arrSize*3];
    float newDist[arrSize*3];
    //float rsi[arrSize*arrSize];
    //float rirjNj[arrSize*arrSize];
    float trueNumCol[arrSize*arrSize];
    //float SrirjNj[arrSize];
    float totCol[arrSize];
    float mass[arrSize];
    float Qds[arrSize];
    float Qe[arrSize*arrSize];
    //float G[arrSize*arrSize];
    //float LRF[arrSize*arrSize];
    float CatColl[arrSize*arrSize];
    float CatCollDis[arrSize*arrSize];
    float CratColl[arrSize*arrSize];
    float CratCollIncreasedLargestObject[arrSize*arrSize];
    float sNumAdded[arrSize];
    float sNumCat[arrSize];
    //float Vesc[arrSize*arrSize];
    float fMesc[arrSize*arrSize];
    float LRFcrat[arrSize*arrSize];
    float Mcrat[arrSize*arrSize];
    float MassJump[2*arrSize];  // this is the mass required to either [crater to a smaller bin,accrete to a larger bin]
    float AccMass[arrSize];
    float MassDiffCrat[arrSize*arrSize]; //this is the mass transfered to a bin from a non-disruptive impact of i and j that isn't incorporated into the parent fragment.
    int accCratBin[arrSize*arrSize]; //this is the bin where the MassDiffCrat mass goes.

    float totMass=0.0;
    
    float rmin; //the minimum radius for fragment distribution

    //for velocity dispersions not equal to impact velocities
    float * velDispSV;
    float * eccSquaredV;
    float * incSquaredV;

    float velDispS[2*arrSize];
    float incSquared[arrSize];
    float eccSquared[arrSize];
    
    //define and allocate the necessary arrays
    CUDA_SAFE_CALL(cudaMalloc((void **) &distV, 3*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &rsiV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &rirjNjV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &rirjNjVCC, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &trueNumColV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &trueNumColIncreasedLargestV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &SrirjNjV, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &totColV, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &QdsV, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &QV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &GV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &LRFV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &CatCollV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &CatCollDisV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &CratCollV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &CratCollIncreasedLargestObjectV, \
			      arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &sNumAddedV, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &sNumCatV, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &VescV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &fMescV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &LRFcratV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &McratV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &AccMassV, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &MassJumpV, 2*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &MassDiffCratV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) & massV, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &QeV, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &numAddV,arrSize*arrSize*arrSize*sizeof(float)));

    //for velocity dispersions not equal to impact velocities
    CUDA_SAFE_CALL(cudaMalloc((void **) &velDispSV, 2*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &incSquaredV, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &eccSquaredV, arrSize*sizeof(float)));

    CUDA_SAFE_CALL(cudaMemset(distV, 0, 3*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(rsiV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(rirjNjV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(rirjNjVCC, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(trueNumColV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(SrirjNjV, 0, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(totColV, 0, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(QdsV, 0, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(QV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(GV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(LRFV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(CatCollV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(CatCollDisV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(CratCollV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(sNumAddedV, 0, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(sNumCatV, 0, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(VescV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(fMescV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(LRFcratV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(McratV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(MassDiffCratV, 0, arrSize*arrSize*sizeof(float)));    
    CUDA_SAFE_CALL(cudaMemset(MassJumpV, 0, 2*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(massV, 0, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(QeV, 0, arrSize*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(AccMassV, 0, arrSize*sizeof(float)));

    //for velocity dispersions not equal to impact velocities
    CUDA_SAFE_CALL(cudaMemset(velDispSV, 0, 2*arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(incSquaredV, 0, arrSize*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(eccSquaredV, 0, arrSize*sizeof(float)));


    
    //initiallize the random number generator
    CUDA_SAFE_CALL(cudaMalloc((void **)&ranValsV, Nran * sizeof(float))); 
    CUDA_SAFE_CALL(cudaMalloc((void **)&ranValsV2, Nran * sizeof(float))); 
    CUDA_SAFE_CALL(cudaMalloc((void **)&ranValsV3, Nran * sizeof(float))); 
    //ranVals = (float *)malloc(Nran * sizeof(float));
    ranVals2 = (float *)malloc(Nran * sizeof(float));
    //ranVals3 = (float *)malloc(Nran * sizeof(float));


    
    ////simulation parameters
    dim3 dimBlock(arrSize);
    dim3 dimGrid(arrSize);
    float megaYear=3.1536e13;
    float gigaYear=3.1536e16;
    float timeStep=megaYear;

    //for Aconstant its only dist[1]
    //for Aipj its only dist*1.e-6
    //for Aij its dist[1]*1.e-6*1.e-6
    //float norm=1.0;


    float mm,am;

    
    simPar simPars;
    ///read in the file parameters
    if (argc>3){
      printf("Reading parameters from file: %s\n",argv[1]);
      useDev=atoi(argv[2]);
      printf("Using CUDA device: %d \n",useDev);
      printf("Restarting from %s.out \n",argv[3]);
      printf("\n");
    } else if (argc>2){
      printf("Reading parameters from file: %s\n",argv[1]);
      useDev=atoi(argv[2]);
      printf("Using CUDA device: %d \n",useDev);
      printf("\n");
    } else if (argc>=1){
      printf("Reading parameters from file: %s\n",argv[1]);
      printf("Using CUDA device: %d \n",useDev);
      
      printf("\n");
    }
    //read in the parameters
    char * outDir=readPars(argv[1],&simPars);
    //initialize the distribution
    char inFileStr[100];
    strcpy(inFileStr,outDir);
    strcat(inFileStr,"/StartDist");
    readStartDist(dist,eccSquared,incSquared,inFileStr,arrSize);
    printf("Starting from the distribution contained in %s.\n",inFileStr);
    for (int i=0;i<arrSize;i++){
      printf("%15.14e %15.14e %15.14e %6.4e %6.4e\n",
	     dist[3*i],dist[3*i+1],dist[3*i+2],
	     eccSquared[i],incSquared[i]);
    }
    printf("\n");
    rmin=dist[0]-dist[2]/2.;


    //set which device the user has chosen
    cudaSetDevice(useDev);
    
    char bunk1[50];
    //char bunk2[50];
    
    printf("a       %f \n",simPars.a);
    printf("bb      %f \n",simPars.bb);
    printf("B       %f \n",simPars.B);
    printf("Qo      %f \n",simPars.Qo);
    printf("v       %f \n",simPars.v);
    printf("r1      %f \n",simPars.r1);
    printf("r2      %f \n",simPars.r2);
    printf("rH      %f \n",simPars.rH);
    printf("qD      %f \n",simPars.qD);
    printf("fQDC    %f \n",simPars.fQDC);
    printf("Gmin    %f \n",simPars.Gmin);
    printf("QeMin   %f \n",simPars.QeMin);
    printf("ad      %f \n",simPars.ad);
    printf("rho     %f \n",simPars.rho);
    printf("tsMax   %f \n",simPars.tsMax);
    printf("tMax    %f \n",simPars.timeMax);
    printf("outDir  %s \n",outDir);
    printf("alpha   %g \n",simPars.alpha);
    printf("fke     %f \n",simPars.fke);
    printf("fshat   %f \n",simPars.fshat);
    printf("qc      %f \n",simPars.qc);
    printf("kcrat   %f \n",simPars.kcrat);
    printf("cratRat %f \n",simPars.maxCratRat);
    printf("q1      %f \n",simPars.q1);
    printf("q2      %f \n",simPars.q2);
    printf("q3      %f \n",simPars.q3);
    printf("q4      %f \n",simPars.q4);
    printf("rb1     %f \n",simPars.rb1);
    printf("rb2     %f \n",simPars.rb2);  
    printf("rb3     %f \n",simPars.rb3);
    printf("outFreq %d \n\n",simPars.outFreq);
  


    float g=pi/((pow(simPars.r2,(float) 2.)-pow(simPars.r1,(float) 2.))*simPars.rH*(pow(1.49E13,3.)));
    float Vmin=pow(2.*simPars.fke/9./simPars.alpha/simPars.fshat,0.5); //minimum ejection velocity set by the cratering parameters
    
    //set the incs,eccs and vels here just to test
    float ao=AU*(simPars.r2+simPars.r1)/2.;
    float area=pi*(powf(AU*simPars.r2,2.0)-powf(AU*simPars.r1,2.0));
    float Omega = powf(Grav*solarMass/ao/ao/ao,0.5f);
    float vK2=ao*ao*Omega*Omega;
    for (int i=0;i<arrSize;i++){
      //eccSquared[i]=1.e-6;
      //incSquared[i]=2.5e-7;
      velDispS[2*i]=(5.0f/8.0f)*eccSquared[i]*vK2;
      velDispS[2*i+1]=0.5f*incSquared[i]*vK2;
    }

    CUDA_SAFE_CALL(cudaMemcpy(distV, dist, 3*arrSize*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(eccSquaredV, eccSquared, arrSize*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(incSquaredV, incSquared, arrSize*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(velDispSV, velDispS, 2*arrSize*sizeof(float), cudaMemcpyHostToDevice));
    


    //if the system were to resume from a previous state
    if (argc>3){
       strcpy(outFileStr,outDir);
       strcat(outFileStr,"/");
       strcat(outFileStr,argv[3]);
       strcat(outFileStr,".out");
       printf(" %s \n",outFileStr);

       runTime=readStep(dist,AccMass,
			eccSquared, incSquared,
			outFileStr,
			arrSize);

       for (int i=0;i<arrSize;i++){
	 printf("%g %g %g %g %g %g \n",log10(dist[i*3]),dist[i*3+1],AccMass[i],eccSquared[i],incSquared[i],totCol[i]);
       }

       outCount=atoi(argv[3])+1;
       whichi=atoi(argv[3])*simPars.outFreq+1;
       printf("runtime %g \n",runTime);
       printf("Starting with iteration %d\n",whichi);
       printf("Next output %d.out \n\n",outCount);
       for (int i=0;i<arrSize;i++){
	 velDispS[2*i]=(5.0f/8.0f)*eccSquared[i]*vK2;
	 velDispS[2*i+1]=0.5f*incSquared[i]*vK2;
       }

       CUDA_SAFE_CALL(cudaMemcpy(distV,dist,3*arrSize*sizeof(float),cudaMemcpyHostToDevice));
       CUDA_SAFE_CALL(cudaMemcpy(AccMassV,AccMass,arrSize*sizeof(float),cudaMemcpyHostToDevice));
       CUDA_SAFE_CALL(cudaMemcpy(incSquaredV,incSquared,arrSize*sizeof(float),cudaMemcpyHostToDevice));
       CUDA_SAFE_CALL(cudaMemcpy(eccSquaredV,eccSquared,arrSize*sizeof(float),cudaMemcpyHostToDevice));
       CUDA_SAFE_CALL(cudaMemcpy(velDispSV,velDispS,2*arrSize*sizeof(float),cudaMemcpyHostToDevice));
       printf("\n");
    }
    
    
    //reset the necessary arrays
    reset2(trueNumCol);
    //reset1(SrirjNj);
    reset1(totCol);


    getMassQdsK<<<1,arrSize>>>(distV,simPars.rho,
			       simPars.Qo,simPars.B,simPars.a,simPars.bb,
			       massV,QdsV);
    getMassJumpK<<<1,arrSize>>>(massV,MassJumpV,arrSize);
    CUDA_SAFE_CALL(cudaMemcpy(mass,massV,arrSize*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(MassJump,MassJumpV,2*arrSize*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(Qds,QdsV,arrSize*sizeof(float),cudaMemcpyDeviceToHost));
    for(int i=0;i<arrSize;i++){
      totMass+=mass[i]*dist[i*3+1];
    }

    ///calculate Vesc(i,j) which is the mutual escape velocity of the two
    /// colliding particles 
    getVescFK<<<arrSize,arrSize>>>(massV,distV,Grav,
				   Vmin,simPars.kcrat,
				   VescV,fMescV,arrSize);
    //CUDA_SAFE_CALL(cudaMemcpy(Vesc,VescV,arrSize*arrSize*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(fMesc,fMescV,arrSize*arrSize*sizeof(float),cudaMemcpyDeviceToHost));

    //copy the dist over to newdist
    for (int i=0;i<arrSize*3;i++){
        newDist[i]=dist[i];
    }



    dim3 blockDim(arrSize);
    dim3 gridDim(arrSize);

    //start of iterations
    for (int ii=whichi;ii<10000000000;ii++){

      
      getQeGK<<<gridDim,blockDim>>>(velDispSV, distV, massV, QdsV, VescV,
				    simPars.rho, simPars.Qo, simPars.B, 
				    simPars.a, simPars.bb, simPars.Gmin, 
				    QeV, GV, arrSize);
      CUDA_SAFE_CALL(cudaMemcpy(Qe,QeV,arrSize*arrSize*sizeof(float),cudaMemcpyDeviceToHost));
      //CUDA_SAFE_CALL(cudaMemcpy(G,GV,arrSize*arrSize*sizeof(float),cudaMemcpyDeviceToHost));

      
      
      //calculate gravitational focusing factors and (ri**2+rj**2)
      rsiK<<<dimGrid,dimBlock>>>(distV,VescV,velDispSV,rsiV,(int) arrSize);
      
      
      //generate the random numbers
      //for collision rounding
      doRandGenFloat(ranValsV, arrSize);
      ulTriangle<<<dimBlock,dimGrid>>>(ranValsV,arrSize);
      
      //for bin reductions
      doRandGenFloat(ranValsV2, arrSize);
      cudaMemcpy(ranVals2,ranValsV2,arrSize*arrSize*sizeof(float),cudaMemcpyDeviceToHost);
      
      //for accretion bin jumping
      doRandGenFloat(ranValsV3, arrSize);

      
      //calculate rirjNj
      rirjNjK<<<dimGrid,dimBlock>>>(rsiV,distV,velDispSV,rirjNjV,arrSize,log10(g));

      
      reduce3(rirjNjV,SrirjNjV,arrSize,arrSize);

      //calculate the timestep considering disruptive collisions only
      //	    return timeStep and number of collisions between bins
      timeStep=bigFunc(distV, rirjNjV, SrirjNjV,
		       massV, QeV, QdsV,
		       GV, VescV, ranValsV,
		       fMescV, numAddV, simPars.QeMin, rmin,
		       simPars.fke, simPars.fshat, simPars.alpha,
		       simPars.qc, simPars.maxCratRat, simPars.rho,
		       simPars.tsMax, arrSize);

      
      runTime+=timeStep;
      if (runTime>simPars.timeMax*megaYear) {
	printf("got to time\n");
	break;}
      
      if ((int) ii/simPars.outFreq == ii/((float ) simPars.outFreq)){
	
	CUDA_SAFE_CALL(cudaMemcpy(totCol,totColV,arrSize*sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(newDist,distV,3*arrSize*sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(AccMass,AccMassV,arrSize*sizeof(float),cudaMemcpyDeviceToHost));
	
	CUDA_SAFE_CALL(cudaMemcpy(eccSquared,eccSquaredV,arrSize*sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(incSquared,incSquaredV,arrSize*sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(velDispS,velDispSV,2*arrSize*sizeof(float),cudaMemcpyDeviceToHost));
	
	printf("\n");
	mm=newDist[1]*mass[0];
	am=AccMass[0];
	for (int i=1;i<arrSize;i++){
	  mm+=mass[i]*newDist[i*3+1];
	  am+=AccMass[i];
	}

	for (int i=0;i<arrSize;i++){
	  
	  printf("%3d %1.6g %1.6g %1.6g %1.6g %1.6g %1.6g %1.6g %1.6g %1.6g %1.6g %1.6g %1.6g %1.6g\n",	\
		 i,
		 log10(dist[i*3]),
		 dist[i*3+1],
		 newDist[i*3+1],
		 sNumAdded[i],
		 sNumCat[i],
		 totCol[i],
		 AccMass[i]/MassJump[2*i],
		 AccMass[i]/MassJump[2*i+1],
		 newDist[i*3+1]/dist[i*3+1],
		 sqrtf(eccSquared[i]),
		 sqrtf(incSquared[i]),
		 sqrtf(velDispS[2*i]),
		 sqrtf(velDispS[2*i+1])
		 );
	  
	}
	printf("%d %g %1.15f     %E %E %1.15f\n",ii,timeStep,runTime/gigaYear,mm/totMass,am/totMass,(mm+am)/totMass);
	
	
	
	//write the distro to file
	strcpy(outFileStr,outDir);
	strcat(outFileStr,"/");
	sprintf(bunk1,"%d",outCount);
	strcat(outFileStr,bunk1);
	strcat(outFileStr, ".out");
	printf("%s \n",outFileStr);
	
	writeStep(newDist,AccMass,totCol,eccSquared,incSquared,runTime,outFileStr,arrSize);

	outCount+=1;
	//if (outCount==10) exit(1);
	
      }
      
      //now reset all arrays to make sure nothing lasts to later
      CUDA_SAFE_CALL(cudaMemset((void *) rirjNjV, 0, arrSize*arrSize*sizeof(float)));
      CUDA_SAFE_CALL(cudaMemset((void *) SrirjNjV, 0, arrSize*sizeof(float)));
      CUDA_SAFE_CALL(cudaMemset((void *) trueNumColV, 0, arrSize*arrSize*sizeof(float)));
      CUDA_SAFE_CALL(cudaMemset((void *) totColV, 0, arrSize*sizeof(float)));
      CUDA_SAFE_CALL(cudaMemset((void *) sNumCatV, 0, arrSize*sizeof(float)));
      
    }
    free(outDir);
}
