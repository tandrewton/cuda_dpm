#ifndef CUDA_DPM_H
#define CUDA_DPM_H

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// pointer-to-member function call macro
#define CALL_MEMBER_FN(object, ptrToMember) ((object).*(ptrToMember))

using namespace std;

__device__ __constant__ int d_NDIM = 2;
__device__ __constant__ int d_numVertices;
__device__ __constant__ double d_rho0;
__device__ __constant__ double d_L[2];
__device__ __constant__ double d_kc;
__device__ __constant__ int d_numVertsPerCell;

class dpm;
typedef void (dpm::*dpmMemFn)(void);

// global constants
const double PI = 4.0 * atan(1.0);
const int nvmin = 12;

// printing constants
const int w = 10;
const int wnum = 25;
const int pnum = 14;

// FIRE constants
const double alpha0 = 0.2;
const double finc = 1.1;
const double fdec = 0.5;
const double falpha = 0.99;

const int NSKIP = 20000;
const int NMIN = 10;
const int NNEGMAX = 1000;
const int NDELAY = 20;
const int itmax = 5e7;

inline __device__ void getVertexPos(const long vId, const double* pos, double* vPos) {
  for (int dim = 0; dim < d_NDIM; dim++) {
    vPos[dim] = pos[vId * d_NDIM + dim];
  }
}

class dpm {
 protected:
  // int scalars
  int NCELLS;
  int NDIM;
  int NNN;
  int NVTOT;
  int vertDOF;

  // time step size
  double dt;

  // potential energy
  double U;
  std::vector<double> cellU;

  // particle spring constants
  double ka;
  double kl;
  double kb;
  double kc;

  // particle attraction constants
  double l1, l2;

  // boundary parameters
  std::vector<double> L;
  std::vector<bool> pbc;

  // particle shape parameters
  std::vector<double> a0;
  std::vector<double> l0;
  std::vector<double> t0;
  std::vector<double> r;

  // indexing variables
  std::vector<int> nv;
  std::vector<int> szList;
  std::vector<int> im1;
  std::vector<int> ip1;

  // dynamical variables
  std::vector<double> x;
  std::vector<double> v;
  std::vector<double> F;
  std::vector<double> vertexEnergy;

  // macroscopic stress vector
  std::vector<double> stress;

  // local stress vector
  std::vector<std::vector<double>> fieldStress;
  std::vector<std::vector<double>> fieldStressCells;

  std::vector<std::vector<double>> fieldShapeStress;
  std::vector<std::vector<double>> fieldShapeStressCells;

  // contact network (vector, size N(N-1)/2), stores # vertex contacts between i-j (i,j are cells)
  // cij is structured as follows: (0-1, 0-2, 0-3, ... ,0- (N-1), 1-2, 1-3, ..., 1- (N-1), 2-3,...)
  std::vector<int> cij;

  // Box linked-list variables
  int NBX;
  std::vector<int> sb;
  std::vector<double> lb;
  std::vector<std::vector<int>> nn;
  std::vector<int> head;
  std::vector<int> last;
  std::vector<int> list;

  // output objects
  std::ofstream posout;

  // cuda variables
  int dimBlock, dimGrid;

 public:
  // Constructors and Destructors
  dpm(int n, int ndim, int seed);
  dpm(int n, int seed)
      : dpm(n, 2, seed) {}
  ~dpm();

  // -- G E T T E R S

  // main ints
  int getNCELLS() { return NCELLS; };
  int getNDIM() { return NDIM; };
  int getNNN() { return NNN; };
  int getNVTOT() { return NVTOT; };
  int getvertDOF() { return vertDOF; };
  int getNV(int ci) { return nv.at(ci); };

  // force parameters
  double getdt() { return dt; };
  double getka() { return ka; };
  double getkl() { return kl; };
  double getkb() { return kb; };
  double getkc() { return kc; };

  // static cell info
  double geta0(int ci) { return a0[ci]; };
  double getl0(int gi) { return l0[gi]; };
  double gett0(int gi) { return t0[gi]; };
  double getr(int gi) { return r[gi]; };

  // dynamic cell info
  double getx(int gi, int d) { return x[NDIM * gi + d]; };
  double getv(int gi, int d) { return v[NDIM * gi + d]; };
  double getF(int gi, int d) { return F[NDIM * gi + d]; };
  double getU() { return U; };

  // boundary variables
  double getL(int d) { return L.at(d); };
  bool getpbc(int d) { return pbc.at(d); };

  // cell shape indexing + information
  int gindex(int ci, int vi);
  void cindices(int& ci, int& vi, int gi);
  double area(int ci);
  double perimeter(int ci);
  void com2D(int ci, double& cx, double& cy);
  double vertexPackingFraction2D();
  double vertexPreferredPackingFraction2D();
  double vertexKineticEnergy();
  int vvContacts();
  int ccContacts();
  void initializeFieldStress();

  // Setters
  void setpbc(int d, bool val) { pbc.at(d) = val; };
  void setNCELLS(int val) { NCELLS = val; };
  void setdt(double val);
  void setka(double val) { ka = val; };
  void setkl(double val) { kl = val; };
  void setkb(double val) { kb = val; };
  void setkc(double val) { kc = val; };
  void setl1(double val) { l1 = val; };
  void setl2(double val) { l2 = val; };
  void scaleL(int d, double val) { L.at(d) *= val; };

  // cuda setters
  void setDeviceVariables(double boxlengthX, double boxlengthY, double density);
  void setBlockGridDims(int dimBlock);
  void cudaVertexNVE(ofstream& enout, double T, double dt0, int NT, int NPRINTSKIP);

  // File openers
  void openPosObject(std::string& str) {
    posout.open(str.c_str());
    if (!posout.is_open()) {
      std::cerr << "	ERROR: posout could not open " << str << "..." << std::endl;
      exit(1);
    } else
      std::cout << "** Opening pos file " << str << " ..." << std::endl;
  }

  // Initialize particles (two dimensions)
  void monodisperse2D(double calA0, int n);
  void bidisperse2D(double calA0, int nsmall, double smallfrac, double sizefrac);
  void gaussian2D(double dispersion, double calA0, int n1);
  void sinusoidalPreferredAngle(double thA, double thK);
  void initializeVertexShapeParameters(double calA0, int nref);
  void initializeVertexShapeParameters(std::vector<double> calA0, int nref);
  void initializeVertexIndexing2D();
  void initializePositions2D(double phi0, double Ftol, bool isFixedBoundary = false, double aspectRatio = 1.0);
  void initializeAllPositions(std::string vertexPositionFile, int nref);
  void initializeFromConfigurationFile(std::string vertexPositionFile, double phi0);
  void initializeNeighborLinkedList2D(double boxLengthScale);

  // editing & updating
  void sortNeighborLinkedList2D();
  void scaleParticleSizes2D(double scaleFactor);
  int removeRattlers();
  void drawVelocities2D(double T);

  // force definitions
  void resetForcesAndEnergy();
  void shapeForces2D();
  void vertexRepulsiveForces2D();
  void vertexAttractiveForces2D();

  // force updates
  void repulsiveForceUpdate();
  void attractiveForceUpdate();

  // simple integrators
  void vertexFIRE2D(dpmMemFn forceCall, double Ftol, double dt0);
  void vertexNVE2D(std::ofstream& enout, dpmMemFn forceCall, double T, double dt0, int NT, int NPRINTSKIP);

  // protocols
  void vertexCompress2Target2D(dpmMemFn forceCall, double Ftol, double dt0, double phi0Target, double dphi0);
  void vertexJamming2D(dpmMemFn forceCall, double Ftol, double Ptol, double dt0, double dphi0, bool plotCompression);

  // print vertex information to file
  void printContactMatrix();
  void printConfiguration2D();
};

#endif
