// File to test dpm forces using NVE protocol
//
// Compilation command:
// g++ -O3 --std=c++11 -I src main/test/dpmNVEtest.cpp src/dpm.cpp -o serialTest.o

// header files
#include "dpm.h"

// preprocessor macros
#define NDIM 2

using namespace std;

int main() {
  // local variables
  int NCELLS = 4, nsmall = 14, seed = 1;
  double phi0 = 0.6, calA0 = 1.2, smallfrac = 0.5, sizefrac = 1.4, Ftol = 1e-12, Ptol = 1e-8, dt0 = 1e-2;
  double ka = 1.0, kl = 1.0, kb = 0.1, kc = 1.0, boxLengthScale = 2.5;

  // pointer to dpm member function (should pt to null)
  dpmMemFn forceUpdate = nullptr;

  // name of output file
  string posf = "pos.test";
  string enf = "en.test";

  // open energy file in main
  ofstream enout(enf.c_str());
  if (!enout.is_open()) {
    cerr << "\t** ERROR: Energy file " << enf << " could not open, ending here." << endl;
    return 1;
  }

  // instantiate object
  dpm configobj2D(NCELLS, NDIM, seed);

  // open position config file
  configobj2D.openPosObject(posf);

  // set spring constants
  configobj2D.setka(ka);
  configobj2D.setkl(kl);
  configobj2D.setkb(kb);
  configobj2D.setkc(kc);

  forceUpdate = &dpm::repulsiveForceUpdate;

  configobj2D.gaussian2D(0.0, calA0, nsmall);

  // initialize particle positions
  configobj2D.initializePositions2D(phi0, Ftol);

  // initialize neighbor linked list
  configobj2D.initializeNeighborLinkedList2D(boxLengthScale);

  // run NVE protocol which will output configuration and energy
  double T = 1e-4;
  /*double ttotal = 100.0;
  double tskip = 10.0;
  int NT = (int)floor(ttotal / dt0);
  int NPRINTSKIP = (int)floor(tskip / dt0);*/
  int NT = 10;
  int NPRINTSKIP = 1;
  configobj2D.vertexNVE2D(enout, forceUpdate, T, dt0, NT, NPRINTSKIP);

  // say goodbye
  cout << "\n\n** Finished dpmNVEtest.cpp, ending. " << endl;

  // close file
  enout.close();

  return 0;
}