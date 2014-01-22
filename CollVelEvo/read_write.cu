#include "read_write.h"
#include <stdio.h>


//input distribution reading prog
void readStartDist(float * dist,
		   float * eccS,
		   float * incS,
		   char * fileName,
		   int Size){
  float r,n,dr,ec,in;

  FILE * fp;
  fp = fopen(fileName,"r");

  for (int i=0;i<Size;i++){
    fscanf(fp,"%f %f %f %f %f",&r,&n,&dr,&ec,&in);
    dist[3*i]=r;
    dist[3*i+1]=n;
    dist[3*i+2]=dr;
    eccS[i]=ec;
    incS[i]=in;
    //printf("%15.14e %15.14e %15.14e %6.4e %6.4e\n",r,n,dr,ec,in);
  }
  fclose(fp);
}
  
//parameter reading prog
char * readPars (char * fileName, simPar * simPars){ 
  FILE *fp;
  fp = fopen(fileName,"r");
  char bunk1[50];
  char bunk2[50];
  static char outDir[8];
  float crap;
  int icrap;

  fscanf(fp,"%s %s",bunk1,bunk2);
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->a = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->bb = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->B = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->Qo = crap;
  
  fscanf(fp,"%s %s",bunk1,&bunk2);
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->v = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->r1 = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->r2 = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->rH = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->qD = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->fQDC = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->Gmin = crap;
  
  
  fscanf(fp,"%s %s",bunk1,bunk2);
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->bmin = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->bratio = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->ad = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->rho = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->tsMax = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->timeMax = crap;
  fscanf(fp,"%s %s",bunk1, outDir);
  
  fscanf(fp,"%s %s",bunk1,bunk2);
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->alpha = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->fke = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->fshat = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->qc = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->kcrat = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->maxCratRat = crap;
  float Rrat=1.-((*simPars).maxCratRat);
  simPars->QeMin=18./10.*(1.-pow(Rrat, (float) 3));
  
  fscanf(fp,"%s %s", bunk1, bunk2);
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->q1 = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->q2 = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->q3 = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->q4 = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->rb1 = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->rb2 = crap;
  fscanf(fp,"%s %f",bunk1, &crap);
  simPars->rb3 = crap;

  fscanf(fp,"%s %s", bunk1,bunk2);
  fscanf(fp,"%s %d",bunk1, &icrap);
  simPars->outFreq = icrap;
  
  fclose(fp);
  return(outDir);

}
  

void writeStep(float * dist,
	       float * accMass,
	       float * totCol,
	       float * eccSquared,
	       float * incSquared,
	       double runTime,
	       char * outFileStr,
	       int Size) {

  FILE * fp=fopen(outFileStr,"w+");
  fprintf(fp,"%1.15f \n",runTime);
  for (int i=0;i<Size;i++){
    fprintf(fp,"%g %g %g %g %g %g \n",log10(dist[i*3]),dist[i*3+1],accMass[i],eccSquared[i],incSquared[i],totCol[i]);
  }
  fclose(fp);
}

double readStep(float * dist, 
	      float * accMass,
	      float * eccSquared,
	      float * incSquared,
	      char * outFileStr,
	      int Size){
  double c0;
  float c1,c2,c3,c4,c5,c6;

  FILE * fp=fopen(outFileStr,"r");
  fscanf(fp,"%lf",&c0);
  //printf("runtime %g \n",c0);

  for (int i=0;i<Size;i++){
    fscanf(fp,"%g %g %g %g %g %g",&c1,&c2,&c3,&c4,&c5,&c6);
    dist[i*3+1] = c2;
    accMass[i] = c3;
    eccSquared[i] = c4;
    incSquared[i]=c5;
  }

  fclose(fp);
  return c0;
}
