
struct simPar {
  //strength parameters
  float a;//=-0.45;
  float bb;//=1.19;
  float B;//=2.1;
  float Qo;//=7e7;
  //impact parameters
  float v;//=1.e5;
  float r1;//=30.;
  float r2;//=60.;
  float rH;//=40.;
  float qD;
  float fQDC;
  float Gmin;//=0.01;
  float QeMin;//=0.4878;
  //miscellany
  float bmin;
  float bratio;//=0.2*50;
  float ad;//=0.2*50;
  float rho;//=1.2;
  float tsMax;//=5.0;
  float timeMax;//=5000.
  float maxCratRat;//=0.1;
  float alpha;//=10e-9; //cratering mass disruption parameter
  float fke;//=0.1;      //fraction of energy into dispersal
  float fshat;//=0.8;   //fraction of energy into shattering
  float qc;//=3.4;     //cratering fragment slope
  float kcrat;//=9./4.; //cratering fragment velocity slope
  //distribution parameters
  float q1;//=4.8;
  float q2;//=2.0;
  float q3;//=3.5;
  float q4;
  float rb1;//=3.e5;
  float rb2;//=1.e4;
  float rb3;
  int outFreq;
};

void readStartDist(float * dist,
		   float * eccS,
		   float * incS,
		   char * fileName,
		   int Size);
char * readPars (char * fileName, simPar * simPars);
void writeStep(float * dist,
	       float * accMass,
	       float * totCol,
	       float * eccSquared,
	       float * incSquared,
	       double runTime,
	       char * outFileStr,
	       int Size);
double readStep(float * dist, 
		float * accMass,
		float * eccSquared,
		float * incSquared,
		char * outFileStr,
		int Size);
