/*

        BASIC FUNCTION DEFINITIONS for DPM class

        Jack Treado, 04/10/21

*/

#include <algorithm>
#include <functional>
#include "cuda_dpm.h"

// namespace
using namespace std;

// cuda kernels
__global__ void kernelVertexForces(double* radius, double* pos, double* force, double* energy) {
  // compare to : vertexRepulsiveForces2D
  /*what does this function need passed into it?
  serial algorithm:
  sort neighbor list
  loop over a reference vertex in the neighbor list
  access neighbor of reference vertex in the neighbor list
  perform distance calculation
  calculate force, energy
  add to force vector
  add to stresses

  parallel algorithm:
  get vertexID
  loop over neighbor list IDs?
  check for neighbors
  call function for two-particle force function

  parallel algorithm 2: don't sort the neighbor list?
  cudaMemCpyToSymbol all of the relevant values like rho0, L[0], L[1], kc
  */
  int vertexID = threadIdx.x + blockDim.x * blockIdx.x;
  int gi = vertexID, ci, cj;
  int NDIM = 2;
  double sij, dx, dy, rij, fx, fy, ftmp;
  energy[NDIM * vertexID] = 0.0;
  ci = gi / d_numVertsPerCell;

  // memCpyToSymbol rho0, L[0], L[1], kc in set

  // printf("vertexID = %d\n", vertexID);
  if (vertexID < d_numVertices) {
    double thisRad, otherRad, interaction = 0;
    double thisPos[2], otherPos[2];
    getVertexPos(vertexID, pos, thisPos);
    thisRad = radius[vertexID];
    // printf("vertexId %d > d_numVertices %d\n", vertexID, d_numVertices);
    // force[vertexID * NDIM] = 1.0;
    // force[vertexID * NDIM + 1] = 2.0;

    for (int gj = 0; gj < d_numVertices; gj++) {
      cj = gj / d_numVertsPerCell;
      if (gi == gj || ci == cj)
        continue;
      // contact distance
      sij = thisRad + radius[gj];
      // particle distance
      dx = pos[NDIM * gj] - pos[NDIM * gi];
      dx -= d_L[0] * round(dx / d_L[0]);
      if (dx < sij) {
        dy = pos[NDIM * gj + 1] - pos[NDIM * gi + 1];
        dy -= d_L[1] * round(dy / d_L[1]);
        if (dy < sij) {
          rij = sqrt(dx * dx + dy * dy);
          if (rij < sij) {
            //  force scale
            ftmp = d_kc * (1 - (rij / sij)) * (d_rho0 / sij);
            fx = ftmp * (dx / rij);
            fy = ftmp * (dy / rij);

            // add to forces
            force[NDIM * gi] -= fx;
            force[NDIM * gi + 1] -= fy;

            // in serial code, would use Newton's second law to cut computation in half. Here, we just go through all particles and don't take advantage of double counting

            // store the energy of half the interaction. the other half comes when we double count gi-gj as gj-gi
            energy[vertexID] += 0.5 * 0.5 * d_kc * pow((1 - (rij / sij)), 2.0);

            printf("gi %d - gj %d at pos %f \t %f, %f \t %f, sij %f, force %f %f\n, ftmp %f, dx %f, rij %f, dU %f\n", vertexID, gj, pos[NDIM * gi], pos[NDIM * gi + 1], pos[NDIM * gj], pos[NDIM * gj + 1], sij, fx, fy, ftmp, dx, rij, 0.5 * 0.5 * d_kc * pow((1 - (rij / sij)), 2.0));
          }
        }
      }
    }
    // printf("total energy = %f\n", energy[vertexID]);
    // printf("force on vertex %d = %f %f\n", vertexID, force[NDIM * gi], force[NDIM * gi + 1]);
  }
}

/******************************

        C O N S T R U C T O R S  &

                D E S T R U C T O R

*******************************/

// Main constructor
dpm::dpm(int n, int ndim, int seed) {
  // local variables
  int d, i;

  // print to console
  cout << "** Instantiating configobj2D object, NCELLS = " << n << ",  ndim = " << ndim << ", seed = " << seed << " ..." << endl;

  // main variables
  NCELLS = n;
  NDIM = ndim;
  NNN = 4;

  // set scalars to default values
  dt = 0.0;

  ka = 0.0;
  kl = 0.0;
  kb = 0.0;
  kc = 0.0;

  l1 = 0.0;
  l2 = 0.0;

  // default boundary variables
  L.resize(NDIM);
  pbc.resize(NDIM);
  for (d = 0; d < NDIM; d++) {
    L[d] = 1.0;
    pbc[d] = 1;
  }

  // preferred area for each cell
  a0.resize(NCELLS);
  cellU.resize(NCELLS);

  // macroscopic stress vector
  stress.resize(NDIM * (NDIM + 1) / 2);
  for (i = 0; i < NDIM * (NDIM + 1) / 2; i++)
    stress.at(i) = 0.0;

  // contact network vector
  cij.resize(NCELLS * (NCELLS - 1) / 2);
  for (i = 0; i < NCELLS * (NCELLS - 1) / 2; i++)
    cij.at(i) = 0;

  // initialize nearest neighbor info
  NBX = -1;

  // seed random number generator
  srand48(seed);
}

// destructor
dpm::~dpm() {
  // clear all private vectors
  // should update this soon
  L.clear();
  pbc.clear();
  a0.clear();
  l0.clear();
  t0.clear();
  nv.clear();
  szList.clear();
  im1.clear();
  ip1.clear();
  r.clear();
  x.clear();
  v.clear();
  F.clear();
  stress.clear();
  sb.clear();
  lb.clear();
  for (int i = 0; i < NBX; i++)
    nn.at(i).clear();
  nn.clear();
  head.clear();
  last.clear();
  list.clear();

  if (posout.is_open())
    posout.close();
}

/******************************

        C E L L   S H A P E

        G E T T E R S

*******************************/

// get global vertex index gi given input cell index ci and vertex index vi
int dpm::gindex(int ci, int vi) {
  return szList[ci] + vi;
}

// get cell index ci and vertex index
void dpm::cindices(int& ci, int& vi, int gi) {
  for (int i = NCELLS - 1; i >= 0; i--) {
    if (gi >= szList[i]) {
      ci = i;
      vi = gi - szList[ci];
      break;
    }
  }
}

// get cell area
double dpm::area(int ci) {
  // local variables
  int vi, vip1, gi, gip1, nvtmp;
  double dx, dy, xi, yi, xip1, yip1, areaVal = 0.0;

  // initial position: vi = 0
  nvtmp = nv.at(ci);
  gi = gindex(ci, 0);
  xi = x[NDIM * gi];
  yi = x[NDIM * gi + 1];

  // loop over vertices of cell ci, get area by shoe-string method
  for (vi = 0; vi < nvtmp; vi++) {
    // next vertex
    gip1 = ip1[gi];
    gi++;

    // get positions (check minimum images)
    dx = x[NDIM * gip1] - xi;
    if (pbc[0])
      dx -= L[0] * round(dx / L[0]);
    xip1 = xi + dx;

    dy = x[NDIM * gip1 + 1] - yi;
    if (pbc[1])
      dy -= L[1] * round(dy / L[1]);
    yip1 = yi + dy;

    // increment area
    areaVal += xi * yip1 - xip1 * yi;

    // set next coordinates
    xi = xip1;
    yi = yip1;
  }
  areaVal *= 0.5;

  return abs(areaVal);
}

// get cell perimeter
double dpm::perimeter(int ci) {
  // local variables
  int vi, gi, gip1, nvtmp;
  double dx, dy, xi, yi, xip1, yip1, l, perimVal = 0.0;

  // initial position: vi = 0
  nvtmp = nv.at(ci);
  gi = gindex(ci, 0);
  xi = x[NDIM * gi];
  yi = x[NDIM * gi + 1];

  // loop over vertices of cell ci, get perimeter
  for (vi = 0; vi < nvtmp; vi++) {
    // next vertex
    gip1 = ip1[gi];
    gi++;

    // get positions (check minimum images)
    dx = x[NDIM * gip1] - xi;
    if (pbc[0])
      dx -= L[0] * round(dx / L[0]);
    xip1 = xi + dx;

    dy = x[NDIM * gip1 + 1] - yi;
    if (pbc[1])
      dy -= L[1] * round(dy / L[1]);
    yip1 = yi + dy;

    // compute segment length
    l = sqrt(dx * dx + dy * dy);

    // add to perimeter
    perimVal += l;

    // update coordinates
    xi = xip1;
    yi = yip1;
  }

  // return perimeter
  return perimVal;
}

// get cell center of mass position
void dpm::com2D(int ci, double& cx, double& cy) {
  // local variables
  int vi, gi, gip1, nvtmp;
  double dx, dy, xi, yi, xip1, yip1, l;

  // initial position: vi = 0
  nvtmp = nv.at(ci);
  gi = gindex(ci, 0);
  xi = x[NDIM * gi];
  yi = x[NDIM * gi + 1];

  // initialize center of mass coordinates
  cx = xi;
  cy = yi;

  // loop over vertices of cell ci, get perimeter
  for (vi = 0; vi < nvtmp - 1; vi++) {
    // next vertex
    gip1 = ip1.at(gi);
    gi++;

    // get positions (check minimum images)
    dx = x[NDIM * gip1] - xi;
    if (pbc[0])
      dx -= L[0] * round(dx / L[0]);
    xip1 = xi + dx;

    dy = x[NDIM * gip1 + 1] - yi;
    if (pbc[1])
      dy -= L[1] * round(dy / L[1]);
    yip1 = yi + dy;

    // add to center of mass
    cx += xip1;
    cy += yip1;

    // update coordinates
    xi = xip1;
    yi = yip1;
  }

  // take average to get com
  cx /= nvtmp;
  cy /= nvtmp;
}

// get configuration packing fraction
double dpm::vertexPackingFraction2D() {
  int ci;
  double val, boxV, areaSum = 0.0;

  // numerator
  for (ci = 0; ci < NCELLS; ci++) {
    areaSum += area(ci) + 0.25 * PI * pow(2.0 * r.at(szList[ci]), 2.0) * (0.5 * nv.at(ci) - 1);
  }

  // denominator
  boxV = L[0] * L[1];

  // return packing fraction
  val = areaSum / boxV;
  return val;
}

// get configuration "preferred" packing fraction
double dpm::vertexPreferredPackingFraction2D() {
  int ci;
  double val, boxV, areaSum = 0.0;

  // numerator
  for (ci = 0; ci < NCELLS; ci++)
    areaSum += a0[ci] + 0.25 * PI * pow(2.0 * r.at(szList[ci]), 2.0) * (0.5 * nv.at(ci) - 1);

  // denominator
  boxV = L[0] * L[1];

  // return packing fraction
  val = areaSum / boxV;
  return val;
}

// get vertex kinetic energy
double dpm::vertexKineticEnergy() {
  double K = 0;

  for (int i = 0; i < vertDOF; i++)
    K += v[i] * v[i];
  K *= 0.5;

  return K;
}

// get number of vertex-vertex contacts
int dpm::vvContacts() {
  int nvv = 0;

  for (int ci = 0; ci < NCELLS; ci++) {
    for (int cj = ci + 1; cj < NCELLS; cj++)
      nvv += cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2];
  }

  return nvv;
}

// get number of cell-cell contacts
int dpm::ccContacts() {
  int ncc = 0;

  for (int ci = 0; ci < NCELLS; ci++) {
    for (int cj = ci + 1; cj < NCELLS; cj++) {
      if (cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2] > 0)
        ncc++;
    }
  }

  return ncc;
}

/******************************

        I N I T I A L -

                        I Z A T I O N

*******************************/

void dpm::initializeFieldStress() {
  // local stress vector
  fieldStress.resize(NVTOT);
  for (int i = 0; i < NVTOT; i++) {
    fieldStress[i].resize(NDIM * (NDIM + 1) / 2);
    for (int j = 0; j < NDIM * (NDIM + 1) / 2; j++)
      fieldStress[i][j] = 0.0;
  }
  fieldShapeStress.resize(NVTOT);
  for (int i = 0; i < NVTOT; i++) {
    fieldShapeStress[i].resize(NDIM * (NDIM + 1) / 2);
    for (int j = 0; j < NDIM * (NDIM + 1) / 2; j++)
      fieldShapeStress[i][j] = 0.0;
  }
  fieldStressCells.resize(NCELLS);
  for (int i = 0; i < NCELLS; i++) {
    fieldStressCells[i].resize(NDIM * (NDIM + 1) / 2);
    for (int j = 0; j < NDIM * (NDIM + 1) / 2; j++)
      fieldStressCells[i][j] = 0.0;
  }
  fieldShapeStressCells.resize(NCELLS);
  for (int i = 0; i < NCELLS; i++) {
    fieldShapeStressCells[i].resize(NDIM * (NDIM + 1) / 2);
    for (int j = 0; j < NDIM * (NDIM + 1) / 2; j++)
      fieldShapeStressCells[i][j] = 0.0;
  }
}

// initialize vertex indexing
void dpm::initializeVertexIndexing2D() {
  int gi, vi, vip1, vim1, ci;

  // check that vertDOF has been assigned
  if (NVTOT <= 0) {
    cerr << "	** ERROR: in initializeVertexIndexing2D, NVTOT not assigned. Need to initialize x, v, and F vectors in this function, so ending here." << endl;
    exit(1);
  }
  if (vertDOF <= 0) {
    cerr << "	** ERROR: in initializeVertexIndexing2D, vertDOF not assigned. Need to initialize x, v, and F vectors in this function, so ending here." << endl;
    exit(1);
  } else if (nv.size() == 0) {
    cerr << "	** ERROR: in initializeVertexIndexing2D, nv vector not assigned. Need to initialize x, v, and F vectors in this function, so ending here." << endl;
    exit(1);
  }

  // save list of adjacent vertices
  im1.resize(NVTOT);
  ip1.resize(NVTOT);
  for (ci = 0; ci < NCELLS; ci++) {
    // vertex indexing
    for (vi = 0; vi < nv.at(ci); vi++) {
      // wrap local indices
      vim1 = (vi - 1 + nv.at(ci)) % nv.at(ci);
      vip1 = (vi + 1) % nv.at(ci);

      // get global wrapped indices
      gi = gindex(ci, vi);
      im1.at(gi) = gindex(ci, vim1);
      ip1.at(gi) = gindex(ci, vip1);
    }
  }

  // initialize vertex configuration vectors
  x.resize(vertDOF);
  v.resize(vertDOF);
  F.resize(vertDOF);
  vertexEnergy.resize(NVTOT);
}

// initialize vertex shape parameters and (a0, l0, t0, r) based on nv (nref is the reference nv, smallest nv among the polydispersity)
void dpm::initializeVertexShapeParameters(double calA0, int nref) {
  // local variables
  int gi, ci, vi, nvtmp;
  double rtmp, calA0tmp, calAntmp;

  // check that vertDOF has been assigned
  if (NVTOT <= 0) {
    cerr << "	** ERROR: in initializeVertexShapeParameters, NVTOT not assigned. Ending here." << endl;
    exit(1);
  }
  if (vertDOF <= 0) {
    cerr << "	** ERROR: in initializeVertexShapeParameters, vertDOF not assigned. Ending here." << endl;
    exit(1);
  } else if (nv.size() == 0) {
    cerr << "	** ERROR: in initializeVertexShapeParameters, nv vector not assigned. Ending here." << endl;
    exit(1);
  }

  // resize shape paramters
  l0.resize(NVTOT);
  t0.resize(NVTOT);
  r.resize(NVTOT);

  // loop over cells, determine shape parameters
  for (ci = 0; ci < NCELLS; ci++) {
    // number of vertices on cell ci
    nvtmp = nv.at(ci);

    // a0 based on nv
    rtmp = (double)nvtmp / nref;
    a0.at(ci) = rtmp * rtmp;

    // shape parameter
    calAntmp = nvtmp * tan(PI / nvtmp) / PI;
    calA0tmp = calA0 * calAntmp;

    // l0 and vertex radii
    gi = szList.at(ci);
    for (vi = 0; vi < nv.at(ci); vi++) {
      l0.at(gi + vi) = 2.0 * sqrt(PI * calA0tmp * a0.at(ci)) / nvtmp;
      t0.at(gi + vi) = 0.0;
      r.at(gi + vi) = 0.5 * l0.at(gi + vi);
    }
  }
}

// initialize vertex shape parameters based on nv (nref is the reference nv, smallest nv among the polydispersity)
void dpm::initializeVertexShapeParameters(std::vector<double> calA0, int nref) {
  // local variables
  int gi, ci, vi, nvtmp;
  double rtmp, calA0tmp, calAntmp;

  // check that vertDOF has been assigned
  if (NVTOT <= 0) {
    cerr << "	** ERROR: in initializeVertexShapeParameters, NVTOT not assigned. Ending here." << endl;
    exit(1);
  }
  if (vertDOF <= 0) {
    cerr << "	** ERROR: in initializeVertexShapeParameters, vertDOF not assigned. Ending here." << endl;
    exit(1);
  } else if (nv.size() == 0) {
    cerr << "	** ERROR: in initializeVertexShapeParameters, nv vector not assigned. Ending here." << endl;
    exit(1);
  }

  // resize shape paramters
  l0.resize(NVTOT);
  t0.resize(NVTOT);
  r.resize(NVTOT);

  // loop over cells, determine shape parameters
  for (ci = 0; ci < NCELLS; ci++) {
    // number of vertices on cell ci
    nvtmp = nv.at(ci);

    // a0 based on nv
    rtmp = (double)nvtmp / nref;
    a0.at(ci) = rtmp * rtmp;

    // shape parameter
    calAntmp = nvtmp * tan(PI / nvtmp) / PI;
    calA0tmp = calA0[ci] * calAntmp;

    // l0 and vertex radii
    gi = szList.at(ci);
    for (vi = 0; vi < nv.at(ci); vi++) {
      l0.at(gi + vi) = 2.0 * sqrt(PI * calA0tmp * a0.at(ci)) / nvtmp;
      t0.at(gi + vi) = 0.0;
      r.at(gi + vi) = 0.5 * l0.at(gi + vi);
    }
  }
}
// initialize monodisperse cell system, single calA0
void dpm::monodisperse2D(double calA0, int n) {
  // local variables
  double calA0tmp, calAntmp, rtmp, areaSum;
  int vim1, vip1, gi, ci, vi, nlarge, smallN, largeN, NVSMALL;

  // print to console
  cout << "** initializing monodisperse DPM particles in 2D ..." << endl;

  // total number of vertices
  NVTOT = n * NCELLS;
  vertDOF = NDIM * NVTOT;

  // szList and nv (keep track of global vertex indices)
  nv.resize(NCELLS);
  szList.resize(NCELLS);

  nv.at(0) = n;
  for (ci = 1; ci < NCELLS; ci++) {
    nv.at(ci) = n;
    szList.at(ci) = szList.at(ci - 1) + nv.at(ci - 1);
  }

  // initialize vertex shape parameters
  initializeVertexShapeParameters(calA0, n);

  // initialize vertex indexing
  initializeVertexIndexing2D();
}

// initialize bidisperse cell system, single calA0
void dpm::bidisperse2D(double calA0, int nsmall, double smallfrac, double sizefrac) {
  // local variables
  double calA0tmp, calAntmp, rtmp, areaSum;
  int vim1, vip1, gi, ci, vi, nlarge, smallN, largeN, NVSMALL;

  // print to console
  cout << "** initializing bidisperse DPM particles in 2D ..." << endl;

  // number of vertices on large particles
  nlarge = round(sizefrac * nsmall);

  // total number of vertices
  smallN = round(smallfrac * NCELLS);
  largeN = NCELLS - smallN;
  NVSMALL = nsmall * smallN;
  NVTOT = NVSMALL + nlarge * largeN;
  vertDOF = NDIM * NVTOT;

  // szList and nv (keep track of global vertex indices)
  nv.resize(NCELLS);
  szList.resize(NCELLS);

  nv.at(0) = nsmall;
  for (ci = 1; ci < NCELLS; ci++) {
    if (ci < smallN) {
      nv.at(ci) = nsmall;
      szList.at(ci) = szList.at(ci - 1) + nv.at(ci - 1);
    } else {
      nv.at(ci) = nlarge;
      szList.at(ci) = szList.at(ci - 1) + nv.at(ci - 1);
    }
  }

  // initialize vertex shape parameters
  initializeVertexShapeParameters(calA0, nsmall);

  // initialize vertex indexing
  initializeVertexIndexing2D();
}

// initialize gaussian polydisperse cell system, single calA0
void dpm::gaussian2D(double dispersion, double calA0, int n1) {
  // local variables
  double calA0tmp, calAntmp, rtmp, areaSum, r1, r2, grv;
  int vim1, vip1, gi, ci, vi, nvtmp;

  // print to console
  cout << "** initializing gaussian DPM particles in 2D with size dispersion " << dispersion << " ..." << endl;

  // szList and nv (keep track of global vertex indices)
  nv.resize(NCELLS);
  szList.resize(NCELLS);

  nv.at(0) = n1;
  NVTOT = n1;
  for (ci = 1; ci < NCELLS; ci++) {
    // use Box-Muller to generate polydisperse sample
    r1 = drand48();
    r2 = drand48();
    grv = sqrt(-2.0 * log(r1)) * cos(2.0 * PI * r2);
    nvtmp = floor(dispersion * n1 * grv + n1);
    if (nvtmp < nvmin)
      nvtmp = nvmin;

    // store size of cell ci
    nv.at(ci) = nvtmp;
    szList.at(ci) = szList.at(ci - 1) + nv.at(ci - 1);

    // add to total NV count
    NVTOT += nvtmp;
  }
  vertDOF = NDIM * NVTOT;

  // initialize vertex shape parameters
  initializeVertexShapeParameters(calA0, n1);

  // initialize vertex indexing
  initializeVertexIndexing2D();
}

// set sinusoidal preferred angle
void dpm::sinusoidalPreferredAngle(double thA, double thK) {
  int ci, vi, gi;
  double thR;

  // print to console
  cout << "** setting initial th0 values to sinusoids, thA = " << thA << ", thK = " << thK << " ..." << endl;

  // loop over cells
  gi = 0;
  for (ci = 0; ci < NCELLS; ci++) {
    thR = (2.0 * PI) / nv.at(ci);
    for (vi = 0; vi < nv.at(ci); vi++) {
      t0.at(gi) = thA * thR * sin(thR * thK * vi);
      gi++;
    }
  }
}

// initialize CoM positions of cells (i.e. use soft disks) using SP FIRE. setupCircularBoundaries enables polygonal walls
void dpm::initializePositions2D(double phi0, double Ftol, bool isFixedBoundary, double aspectRatio) {
  // isFixedBoundary is an optional bool argument that tells cells to stay away from the boundary during initialization
  // aspectRatio is the ratio L[0] / L[1]
  int i, d, ci, cj, vi, vj, gi, cellDOF = NDIM * NCELLS;
  int numEdges = 20;  // number of edges in the polygonal walls to approximate a circle
  double areaSum, xtra = 1.1;
  std::vector<double> aspects = {1.0 * aspectRatio, 1.0 / aspectRatio};

  // local disk vectors
  vector<double> drad(NCELLS, 0.0);
  vector<double> dpos(cellDOF, 0.0);
  vector<double> dv(cellDOF, 0.0);
  vector<double> dF(cellDOF, 0.0);

  // print to console
  cout << "** initializing particle positions using 2D SP model and FIRE relaxation ..." << endl;

  // initialize stress field
  initializeFieldStress();

  // initialize box size based on packing fraction
  areaSum = 0.0;
  for (ci = 0; ci < NCELLS; ci++)
    areaSum += a0.at(ci) + 0.25 * PI * pow(l0.at(ci), 2.0) * (0.5 * nv.at(ci) - 1);

  // set box size : phi_0 = areaSum / A => A = areaSum/phi_0 which gives us the following formulas for L
  for (d = 0; d < NDIM; d++) {
    L.at(d) = pow(areaSum / phi0, 1.0 / NDIM) * aspects[d];
  }

  // initialize cell centers randomly
  for (ci = 0; ci < cellDOF; ci += 2) {
    dpos.at(ci) = L[ci % 2] * drand48();
  }
  for (ci = cellDOF - 1; ci > 0; ci -= 2) {
    dpos.at(ci) = L[ci % 2] * drand48();
  }

  // set radii of SP disks
  for (ci = 0; ci < NCELLS; ci++) {
    drad.at(ci) = sqrt((2.0 * a0.at(ci)) / (nv.at(ci) * sin(2.0 * PI / nv.at(ci))));
  }

  // FIRE VARIABLES
  double P = 0;
  double fnorm = 0;
  double vnorm = 0;
  double alpha = alpha0;

  double dt0 = 1e-2;
  double dtmax = 10 * dt0;
  double dtmin = 1e-8 * dt0;

  int npPos = 0;
  int npNeg = 0;

  int fireit = 0;
  double fcheck = 10 * Ftol;

  // interaction variables
  double rij, sij, dtmp, ftmp, vftmp;
  double dr[NDIM];

  // initial step size
  dt = dt0;

  // loop until force relaxes
  while ((fcheck > Ftol) && fireit < itmax) {
    // FIRE step 1. Compute P
    P = 0.0;
    for (i = 0; i < cellDOF; i++)
      P += dv[i] * dF[i];

    // FIRE step 2. adjust simulation based on net motion of degrees of freedom
    if (P > 0) {
      // increase positive counter
      npPos++;

      // reset negative counter
      npNeg = 0;

      // alter simulation if enough positive steps have been taken
      if (npPos > NMIN) {
        // change time step
        if (dt * finc < dtmax)
          dt *= finc;

        // decrease alpha
        alpha *= falpha;
      }
    } else {
      // reset positive counter
      npPos = 0;

      // increase negative counter
      npNeg++;

      // check if simulation is stuck
      if (npNeg > NNEGMAX) {
        cerr << "	** ERROR: During initial FIRE minimization, P < 0 for too long, so ending." << endl;
        exit(1);
      }

      // take half step backwards, reset velocities
      for (i = 0; i < cellDOF; i++) {
        // take half step backwards
        dpos[i] -= 0.5 * dt * dv[i];

        // reset velocities
        dv[i] = 0.0;
      }

      // decrease time step if past initial delay
      if (fireit > NDELAY) {
        // decrease time step
        if (dt * fdec > dtmin)
          dt *= fdec;

        // reset alpha
        alpha = alpha0;
      }
    }

    // FIRE step 3. First VV update
    for (i = 0; i < cellDOF; i++)
      dv[i] += 0.5 * dt * dF[i];

    // FIRE step 4. adjust velocity magnitude
    fnorm = 0.0;
    vnorm = 0.0;
    for (i = 0; i < cellDOF; i++) {
      fnorm += dF[i] * dF[i];
      vnorm += dv[i] * dv[i];
    }
    fnorm = sqrt(fnorm);
    vnorm = sqrt(vnorm);
    if (fnorm > 0) {
      for (i = 0; i < cellDOF; i++)
        dv[i] = (1 - alpha) * dv[i] + alpha * (vnorm / fnorm) * dF[i];
    }

    // FIRE step 4. Second VV update
    for (i = 0; i < cellDOF; i++) {
      dpos[i] += dt * dv[i];
      dF[i] = 0.0;
    }

    // FIRE step 5. Update forces
    for (ci = 0; ci < NCELLS; ci++) {
      for (cj = ci + 1; cj < NCELLS; cj++) {
        // contact distance
        sij = drad[ci] + drad[cj];

        // true distance
        rij = 0.0;
        for (d = 0; d < NDIM; d++) {
          // get distance element
          dtmp = dpos[NDIM * cj + d] - dpos[NDIM * ci + d];
          if (pbc[d])
            dtmp -= L[d] * round(dtmp / L[d]);

          // add to true distance
          rij += dtmp * dtmp;

          // save in distance array
          dr[d] = dtmp;
        }
        rij = sqrt(rij);

        // check distances
        if (rij < sij) {
          // force magnitude
          ftmp = kc * (1.0 - (rij / sij)) / sij;

          // add to vectorial force
          for (d = 0; d < NDIM; d++) {
            vftmp = ftmp * (dr[d] / rij);
            dF[NDIM * ci + d] -= vftmp;
            dF[NDIM * cj + d] += vftmp;
          }
        }
      }
    }
    // FIRE step 4.1 Compute wall forces
    if (isFixedBoundary) {
      for (i = 0; i < cellDOF; i++) {
        bool collideTopOrRight = dpos[i] > L[i % NDIM] - drad[i];
        bool collideBottomOrLeft = dpos[i] < drad[i];

        if (collideTopOrRight) {  // deflect particle down or left
          dF[i] += -1 * (drad[i] - L[i % NDIM] + dpos[i]);
        }
        if (collideBottomOrLeft) {
          dF[i] += 1 * (drad[i] - dpos[i]);
        }
      }
    }

    // FIRE step 5. Final VV update
    for (i = 0; i < cellDOF; i++)
      dv[i] += 0.5 * dt * dF[i];

    // update forces to check
    fcheck = 0.0;
    for (i = 0; i < cellDOF; i++)
      fcheck += dF[i] * dF[i];
    fcheck = sqrt(fcheck / NCELLS);

    // print to console
    if (fireit % NSKIP == 0) {
      cout << endl
           << endl;
      cout << "===========================================" << endl;
      cout << "		I N I T I A L  S P 			" << endl;
      cout << " 	F I R E 						" << endl;
      cout << "		M I N I M I Z A T I O N 	" << endl;
      cout << "===========================================" << endl;
      cout << endl;
      cout << "	** fireit = " << fireit << endl;
      cout << "	** fcheck = " << fcheck << endl;
      cout << "	** fnorm = " << fnorm << endl;
      cout << "	** vnorm = " << vnorm << endl;
      cout << "	** dt = " << dt << endl;
      cout << "	** P = " << P << endl;
      cout << "	** Pdir = " << P / (fnorm * vnorm) << endl;
      cout << "	** alpha = " << alpha << endl;
    }

    // update iterate
    fireit++;
  }
  // check if FIRE converged
  if (fireit == itmax) {
    cout << "	** FIRE minimization did not converge, fireit = " << fireit << ", itmax = " << itmax << "; ending." << endl;
    exit(1);
  } else {
    cout << endl
         << endl;
    cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;
    cout << "===========================================" << endl;
    cout << " 	F I R E 						" << endl;
    cout << "		M I N I M I Z A T I O N 	" << endl;
    cout << "	C O N V E R G E D! 				" << endl
         << endl;

    cout << "	(for initial disk minimization) " << endl;
    cout << "===========================================" << endl;
    cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;
    cout << endl;
    cout << "	** fireit = " << fireit << endl;
    cout << "	** fcheck = " << fcheck << endl;
    cout << "	** vnorm = " << vnorm << endl;
    cout << "	** dt = " << dt << endl;
    cout << "	** P = " << P << endl;
    cout << "	** alpha = " << alpha << endl;
  }

  // initialize vertex positions based on cell centers
  for (ci = 0; ci < NCELLS; ci++) {
    for (vi = 0; vi < nv.at(ci); vi++) {
      // get global vertex index
      gi = gindex(ci, vi);

      // length from center to vertex
      dtmp = sqrt((2.0 * a0.at(ci)) / (nv.at(ci) * sin((2.0 * PI) / nv.at(ci))));

      // set positions
      x.at(NDIM * gi) = dtmp * cos((2.0 * PI * vi) / nv.at(ci)) + dpos.at(NDIM * ci) + 1e-2 * l0[gi] * drand48();
      x.at(NDIM * gi + 1) = dtmp * sin((2.0 * PI * vi) / nv.at(ci)) + dpos.at(NDIM * ci + 1) + 1e-2 * l0[gi] * drand48();
    }
  }
}

void dpm::initializeFromConfigurationFile(std::string vertexPositionFile, double phi0) {
  // in case of variable calA0, nv, and positions, this function subsumes monodisperse2D

  std::ifstream positionFile(vertexPositionFile);
  int cellNum, vertNum, a, b;
  double vertLocX, vertLocY;
  std::vector<int> numVertsPerDPM;
  std::vector<double> vertexPositions;

  while (positionFile >> cellNum >> vertNum) {  // header line is cell ID and total # vertices
    numVertsPerDPM.push_back(vertNum);
    for (int i = 0; i < vertNum; i++) {
      positionFile >> a >> b >> vertLocX >> vertLocY;
      vertexPositions.push_back(vertLocX);
      vertexPositions.push_back(vertLocY);
    }
  }

  // total number of vertices
  NCELLS = cellNum;
  cout << "NCELLS = " << cellNum << '\n';

  NVTOT = 0;
  for (auto i : numVertsPerDPM) {
    NVTOT += i;
  }
  vertDOF = NDIM * NVTOT;

  // szList and nv keep track of global vertex indices
  nv.resize(NCELLS);
  szList.resize(NCELLS);

  nv.at(0) = numVertsPerDPM[0];
  for (int ci = 1; ci < NCELLS; ci++) {
    nv.at(ci) = numVertsPerDPM[ci];
    szList.at(ci) = szList.at(ci - 1) + nv.at(ci - 1);
  }

  // initialize connectivity between vertices in DPs
  initializeVertexIndexing2D();

  // minimum coordinates to subtract so particles start near origin
  double min_x = 1e10, min_y = 1e10;
  for (int i = 0; i < vertexPositions.size(); i += 2) {
    if (vertexPositions[NDIM * i] < min_x)
      min_x = vertexPositions[NDIM * i];
    if (vertexPositions[NDIM * i + 1] < min_y)
      min_y = vertexPositions[NDIM * i + 1];
  }

  int gi;
  // initialize vertex positions
  for (int ci = 0; ci < NCELLS; ci++) {
    for (int vi = 0; vi < nv.at(ci); vi++) {
      gi = gindex(ci, vi);
      x.at(gi * NDIM) = vertexPositions[gi * NDIM] - min_x;
      x.at(gi * NDIM + 1) = vertexPositions[gi * NDIM + 1] - min_y;
    }
  }

  // once i've assigned cell IDs and vertex coordinates, use dpm functions to set the preferred lengths, radii

  // resize shape paramters
  l0.resize(NVTOT);
  t0.resize(NVTOT);
  r.resize(NVTOT);

  double l0_all, r_all, d_all;  // variables for rescaling lengths
  // loop over cells, determine shape parameters
  for (int ci = 0; ci < NCELLS; ci++) {
    double l0_ci, r_ci;
    for (int vi = 0; vi < nv.at(ci); vi++) {
      gi = gindex(ci, vi);
      // l0.at(gi) = sqrt(pow(x.at(NDIM * ip1[gi]) - x.at(NDIM * gi), 2) + pow(x.at(NDIM * ip1[gi] + 1) - x.at(NDIM * gi + 1), 2));
      // save parameters from the first vertex of each cell to set l0, r
      if (vi == 0) {
        l0_ci = sqrt(pow(x.at(NDIM * ip1[gi]) - x.at(NDIM * gi), 2) + pow(x.at(NDIM * ip1[gi] + 1) - x.at(NDIM * gi + 1), 2));
        r_ci = 0.5 * l0_ci;
        if (ci == 0) {
          // save parameters from the first vertex of the first cell to set length renormalization scale
          l0_all = l0_ci;
          r_all = l0_ci / 2.0;
          d_all = l0_ci;
        }
      }
      l0.at(gi) = l0_ci / d_all;
      t0.at(gi) = 0.0;
      r.at(gi) = r_ci / d_all;
      cout << "r[gi] at time of setting = " << r[gi] << '\n';
      x[NDIM * gi] /= d_all;
      x[NDIM * gi + 1] /= d_all;
    }
  }

  for (int d = 0; d < NDIM; d++) {
    L.at(d) = 1e10;  // set L to be large to not interfere with initial shape calculations, reset L later
  }

  double areaSum = 0.0;
  for (int ci = 0; ci < NCELLS; ci++) {
    areaSum += area(ci);
    a0.at(ci) = area(ci);
    cout << "new shape subject to length rescaling is = " << pow(perimeter(ci), 2) / (4 * 3.1415 * area(ci)) << '\n';
  }
  // todo: calculate area of all DP cells, then use input phi to give myself a square box. then run testInterpolatedConfiguration.m to see the initial configuration and judge its quality

  // set box size
  for (int d = 0; d < NDIM; d++) {
    // L.at(d) = pow(areaSum / phi0, 1.0 / NDIM);
    double max = *max_element(x.begin(), x.end());
    L.at(d) = 1.1 * max;
  }
}

// initialize vertices according to an input configuration file, and initialize/set relevant lists and degrees of freedom variables.
void dpm::initializeAllPositions(std::string vertexPositionFile, int nref) {
  // for every block in vertexPositionFile, place vertices according to a particle.
  // vertexPositionFile should have NCELLS blocks separated by headers
  // each block should have a header
  //      'nan nan'
  //      'N(particle label) NV(number of vertices, raw)'
  //
  // every successive line should have x and y positions for vertices belonging to the cell
  // every line (x, y) corresponds to a vertex position, spaced 1-pixel away from the previous line
  // start by calculating the perimeter, area, shape of the block

  double posx, posy;
  double pixelsPerParticle;
  int N = -1, numVertsCounter = 0, lineCounter = 0;
  std::vector<double> vertexPositions;
  std::vector<int> numVerts(NCELLS);
  std::vector<double> shapes;

  // read in file
  ifstream positionFile(vertexPositionFile);
  while (positionFile >> posx >> posy) {
    if (std::isnan(posx)) {  // found a header, so read in the next line and calculate how many vertices we want
      N++;
      positionFile >> posx >> posy;  // particle label and pixels in perimeter
      if (N == 0) {
        pixelsPerParticle = posy / double(nref);  // since N = 0 corresponds to the smallest perimeter, set this as baseline
      }
      if (N > 0) {  // need to store vertexPositions and other cell related quantities, since we are moving to a new cell
        numVerts.push_back(numVertsCounter);
      }
      // reset counters
      numVertsCounter = 0;
      lineCounter = 0;
      continue;
    }
    if (lineCounter == round(pixelsPerParticle)) {  // we have read in the coordinates of the vertex we want to initialize
      vertexPositions.push_back(posx);
      vertexPositions.push_back(posy);
      numVertsCounter++;
    }
    lineCounter++;
  }
  double perimeter = 0;
  double area = 0;

  for (int i = 0; i < N; i++) {
    // calculate perimeter
    // calculate area
    // shapes.push_back(perimeter^2/4pi*area);
  }

  // might need to set box length at some point here

  // set NVTOT, vertDOF, nv, szList, vertex lists
  // NCELLS = N;
  // NVTOT = accumulate(numVerts);
  // vertDOF = NVTOT * NDIM;
  // nv.resize(NCELLS);
  // szList.resize(NCELLS);
  // nv.at(0) = nref?;
  // for (ci = 1; ci < NCELLS; ci++){
  //  nv.at(ci) = nref;
  //  szList.at(ci) = szList.at(ci-1) + nv.at(ci-1);
  // }
  //

  initializeVertexShapeParameters(shapes, nref);  // overloaded this function, need to check later if it compiles

  initializeVertexIndexing2D();

  int gi;
  // initialize vertex positions
  for (int ci = 0; ci < NCELLS; ci++) {
    for (int vi = 0; vi < nv.at(ci); vi++) {
      gi = gindex(ci, vi);
      x.at(gi * NDIM) = vertexPositions[gi * NDIM];  // not sure about this, double check RHS
      x.at(gi * NDIM + 1) = vertexPositions[gi * NDIM + 1];
    }
  }
}

// initialize neighbor linked list
void dpm::initializeNeighborLinkedList2D(double boxLengthScale) {
  // local variables
  double llscale;
  int i, d, nntmp, scx;

  // print to console
  cout << "** initializing neighbor linked list, boxLengthScale = " << boxLengthScale << '\n';

  // get largest diameter times attraction shell (let buffer be larger than attraction range would ever be) as llscale
  double buffer = 2.5;
  llscale = buffer * 2 * (*max_element(r.begin(), r.end()));
  cout << "llscale = " << llscale << '\n';

  // initialize box length vectors
  NBX = 1;
  sb.resize(NDIM);
  lb.resize(NDIM);
  for (d = 0; d < NDIM; d++) {
    // determine number of cells along given dimension by rmax
    sb[d] = round(L[d] / (boxLengthScale * llscale));

    // just in case, if < 3, change to 3 so box neighbor checking will work
    if (sb[d] < 3)
      sb[d] = 3;

    // determine box length by number of cells
    lb[d] = L[d] / sb[d];

    // count total number of cells
    NBX *= sb[d];
  }

  // initialize list of box nearest neighbors
  scx = sb[0];
  nn.resize(NBX);

  // loop over cells, save forward neighbors for each box
  for (i = 0; i < NBX; i++) {
    // reshape entry
    nn[i].resize(NNN);

    // right neighbor (i+1)
    nn[i][0] = (i + 1) % NBX;

    // top neighbors (i,j+1), (i+1,j+1)
    if (pbc[1]) {
      // (i,j+1) w/ pbc
      nn[i][1] = (i + scx) % NBX;

      // (i+1,j+1) w/ pbc
      nn[i][2] = (nn[i][1] + 1) % NBX;
    } else {
      // if on top row, both = -1
      if (i >= NBX - scx) {
        nn[i][1] = -1;
        nn[i][2] = -1;
      }
      // if not on top row, still add
      else {
        nn[i][1] = i + scx;
        nn[i][2] = nn[i][1] + 1;
      }
    }

    // bottom neighbor w/ pbc (j-1)
    nntmp = (i + NBX - scx) % NBX;

    // bottom-right neighbor (i+1, j-1)
    if (pbc[1])
      nn[i][3] = nntmp + 1;
    else {
      // if on bottom row, skip
      if (i < scx)
        nn[i][3] = -1;
      // otherwise, set
      else
        nn[i][3] = nntmp + 1;
    }

    // right-hand bc (periodic)
    if ((i + 1) % scx == 0) {
      if (pbc[0]) {
        nn[i][0] = i - scx + 1;
        if (pbc[1]) {
          nn[i][2] = nn[i][1] - scx + 1;
          nn[i][3] = nntmp - scx + 1;
        }
      } else {
        nn[i][0] = -1;
        nn[i][2] = -1;
        nn[i][3] = -1;
      }
    }
  }

  // linked-list variables
  head.resize(NBX);
  last.resize(NBX);
  list.resize(NVTOT + 1);

  // print box info to console
  cout << ";  initially NBX = " << NBX << " ..." << endl;
}

/******************************

        E D I T I N G   &

                        U P D A T I N G

*******************************/

// sort vertices into neighbor linked list
void dpm::sortNeighborLinkedList2D() {
  // local variables
  int d, gi, boxid, sbtmp;
  double xtmp;

  /*cout << "before neighborLinkedList\n";
  cout << "list = \n";
  for (auto i : list)
    cout << i << '\t';
  cout << "\nhead = \n";
  for (auto i : head)
    cout << i << '\t';
  cout << "\nlast = \n";
  for (auto i : last)
    cout << i << '\t';*/

  // reset linked list info
  fill(list.begin(), list.end(), 0);
  fill(head.begin(), head.end(), 0);
  fill(last.begin(), last.end(), 0);
  // sort vertices into linked list
  for (gi = 0; gi < NVTOT; gi++) {
    // 1. get cell id of current particle position
    boxid = 0;
    sbtmp = 1;
    for (d = 0; d < NDIM; d++) {
      // current location
      xtmp = x[NDIM * gi + d];
      // check out-of-bounds
      if (xtmp < 0) {
        if (pbc[d])
          xtmp -= L[d] * floor(xtmp / L[d]);
        else
          xtmp = 0.00001;
      } else if (xtmp > L[d]) {
        if (pbc[d])
          xtmp -= L[d] * floor(xtmp / L[d]);
        else
          xtmp = 0.99999 * L[d];
      }

      // add d index to 1d list
      boxid += floor(xtmp / lb[d]) * sbtmp;
      if (boxid < -2147483600) {
        cout << "boxid = " << boxid << ", xtmp = " << xtmp << ", lb[d] = " << lb[d] << ", d = " << d << ", gi = " << gi << ", floor(xtmp / lb[d]) = " << floor(xtmp / lb[d]) << ", list.size = " << list.size() << '\n';
        cout << "pbc[d] = " << pbc[d] << ", L[d] = " << L[d] << '\n';

        for (int k = 0; k < NVTOT; k++) {
          cout << "vert " << k << ": " << x[NDIM * k] << ", " << x[NDIM * k + 1] << '\n';
        }
      }

      // increment dimensional factor
      sbtmp *= sb[d];
    }

    // 2. add to head list or link within list
    // NOTE: particle ids are labelled starting from 1, setting to 0 means end of linked list
    if (head[boxid] == 0) {
      head[boxid] = gi + 1;
      last[boxid] = gi + 1;
    } else {
      list[last[boxid]] = gi + 1;
      last[boxid] = gi + 1;
    }
  }
  /*cout << "after neighborLinkedList\n";
  cout << "list = \n";
  for (auto i : list)
    cout << i << '\t';
  cout << "\nhead = \n";
  for (auto i : head)
    cout << i << '\t';
  cout << "\nlast = \n";
  for (auto i : last)
    cout << i << '\t';*/
}

// change size of particles
void dpm::scaleParticleSizes2D(double scaleFactor) {
  // local variables
  int gi, ci, vi, xind, yind;
  double xi, yi, cx, cy, dx, dy;

  // loop over cells, scale
  for (ci = 0; ci < NCELLS; ci++) {
    // scale preferred area
    a0[ci] *= scaleFactor * scaleFactor;

    // first global index for ci
    gi = szList.at(ci);

    // compute cell center of mass
    xi = x[NDIM * gi];
    yi = x[NDIM * gi + 1];
    cx = xi;
    cy = yi;
    for (vi = 1; vi < nv.at(ci); vi++) {
      dx = x.at(NDIM * (gi + vi)) - xi;
      if (pbc[0])
        dx -= L[0] * round(dx / L[0]);

      dy = x.at(NDIM * (gi + vi) + 1) - yi;
      if (pbc[1])
        dy -= L[1] * round(dy / L[1]);

      xi += dx;
      yi += dy;

      cx += xi;
      cy += yi;
    }
    cx /= nv.at(ci);
    cy /= nv.at(ci);

    for (vi = 0; vi < nv.at(ci); vi++) {
      // x and y inds
      xind = NDIM * (gi + vi);
      yind = xind + 1;

      // closest relative position
      dx = x[xind] - cx;
      if (pbc[0])
        dx -= L[0] * round(dx / L[0]);

      dy = x[yind] - cy;
      if (pbc[1])
        dy -= L[1] * round(dy / L[1]);

      // update vertex positions
      x[xind] += (scaleFactor - 1.0) * dx;
      x[yind] += (scaleFactor - 1.0) * dy;

      // scale vertex radii
      r[gi + vi] *= scaleFactor;
      l0[gi + vi] *= scaleFactor;
    }
  }
}

// remove rattlers from contact network, return number of rattlers
int dpm::removeRattlers() {
  // local variables
  int ci, cj, ctmp, rvv, rcc, nr, nm = 1;

  // loop over rows, eliminate contacts to rattlers until nm = 0
  while (nm > 0) {
    // reset number of rattlers
    nr = 0;

    // number of "marginal" rattlers to be removed
    nm = 0;
    for (ci = 0; ci < NCELLS; ci++) {
      // get number of contacts on cell ci
      rvv = 0;
      rcc = 0;
      for (cj = 0; cj < NCELLS; cj++) {
        if (ci != cj) {
          if (ci > cj)
            ctmp = cij[NCELLS * cj + ci - (cj + 1) * (cj + 2) / 2];
          else
            ctmp = cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2];
        } else
          ctmp = 0;

        rvv += ctmp;
        if (ctmp > 0)
          rcc++;
      }

      // check to see if particle should be removed from network
      if (rcc <= NDIM && rvv <= 3) {
        // increment # of rattlers
        nr++;

        // if in contact, remove contacts
        if (rvv > 0) {
          nm++;

          for (cj = 0; cj < NCELLS; cj++) {
            // delete contact between ci and cj
            if (ci != cj) {
              if (ci > cj)
                cij[NCELLS * cj + ci - (cj + 1) * (cj + 2) / 2] = 0;
              else
                cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2] = 0;
            }
          }
        }
      }
    }
  }

  // return total number of rattlers
  return nr;
}

// draw random velocities based on input temperature
void dpm::drawVelocities2D(double T) {
  // local variables
  int gi;
  double r1, r2, grv1, grv2, tscale = sqrt(T), vcomx = 0.0, vcomy = 0.0;

  // loop over velocities, draw from maxwell-boltzmann distribution
  for (gi = 0; gi < NVTOT; gi++) {
    // draw random numbers using Box-Muller
    r1 = drand48();
    r2 = drand48();
    grv1 = sqrt(-2.0 * log(r1)) * cos(2.0 * PI * r2);
    grv2 = sqrt(-2.0 * log(r1)) * sin(2.0 * PI * r2);

    // assign to velocities
    v[NDIM * gi] = tscale * grv1;
    v[NDIM * gi + 1] = tscale * grv2;

    // add to center of mass
    vcomx += v[NDIM * gi];
    vcomy += v[NDIM * gi + 1];
  }
  vcomx = vcomx / NVTOT;
  vcomy = vcomy / NVTOT;

  // subtract off center of mass drift
  for (gi = 0; gi < NVTOT; gi++) {
    v[NDIM * gi] -= vcomx;
    v[NDIM * gi + 1] -= vcomy;
  }
}

/******************************

        D P M  F O R C E

                        U P D A T E S

*******************************/

void dpm::resetForcesAndEnergy() {
  fill(F.begin(), F.end(), 0.0);
  fill(stress.begin(), stress.end(), 0.0);
  fill(fieldStress.begin(), fieldStress.end(), vector<double>(3, 0.0));
  fill(fieldShapeStress.begin(), fieldShapeStress.end(), vector<double>(3, 0.0));
  fill(fieldStressCells.begin(), fieldStressCells.end(), vector<double>(3, 0.0));
  fill(fieldShapeStressCells.begin(), fieldShapeStressCells.end(), vector<double>(3, 0.0));
  U = 0.0;
  fill(cellU.begin(), cellU.end(), 0.0);
}

void dpm::shapeForces2D() {
  // local variables
  int ci, gi, vi, nvtmp;
  double fa, fli, flim1, fb, cx, cy, xi, yi;
  double rho0, l0im1, l0i, a0tmp, atmp;
  double dx, dy, da, dli, dlim1, dtim1, dti, dtip1;
  double lim2x, lim2y, lim1x, lim1y, lix, liy, lip1x, lip1y, li, lim1;
  double rim2x, rim2y, rim1x, rim1y, rix, riy, rip1x, rip1y, rip2x, rip2y;
  double nim1x, nim1y, nix, niy, sinim1, sini, sinip1, cosim1, cosi, cosip1;
  double ddtim1, ddti;
  double forceX, forceY;
  double unwrappedX, unwrappedY;

  // loop over vertices, add to force
  rho0 = sqrt(a0.at(0));
  ci = 0;
  for (gi = 0; gi < NVTOT; gi++) {
    // -- Area force (and get cell index ci)
    if (ci < NCELLS) {
      if (gi == szList[ci]) {
        // shape information
        nvtmp = nv[ci];
        a0tmp = a0[ci];

        // preferred segment length of last segment
        l0im1 = l0[im1[gi]];

        // compute area deviation
        atmp = area(ci);
        da = (atmp / a0tmp) - 1.0;

        // update potential energy
        U += 0.5 * ka * (da * da);
        cellU[ci] += 0.5 * ka * (da * da);

        // shape force parameters
        fa = ka * da * (rho0 / a0tmp);
        fb = kb * rho0;

        // compute cell center of mass
        xi = x[NDIM * gi];
        yi = x[NDIM * gi + 1];
        cx = xi;
        cy = yi;
        for (vi = 1; vi < nvtmp; vi++) {
          // get distances between vim1 and vi
          dx = x[NDIM * (gi + vi)] - xi;
          dy = x[NDIM * (gi + vi) + 1] - yi;
          if (pbc[0])
            dx -= L[0] * round(dx / L[0]);
          if (pbc[1])
            dy -= L[1] * round(dy / L[1]);

          // add to centers
          xi += dx;
          yi += dy;

          cx += xi;
          cy += yi;
        }
        cx /= nvtmp;
        cy /= nvtmp;

        // get coordinates relative to center of mass
        rix = x[NDIM * gi] - cx;
        riy = x[NDIM * gi + 1] - cy;

        // get prior adjacent vertices
        rim2x = x[NDIM * im1[im1[gi]]] - cx;
        rim2y = x[NDIM * im1[im1[gi]] + 1] - cy;
        if (pbc[0])
          rim2x -= L[0] * round(rim2x / L[0]);
        if (pbc[1])
          rim2y -= L[1] * round(rim2y / L[1]);

        rim1x = x[NDIM * im1[gi]] - cx;
        rim1y = x[NDIM * im1[gi] + 1] - cy;
        if (pbc[0])
          rim1x -= L[0] * round(rim1x / L[0]);
        if (pbc[1])
          rim1y -= L[1] * round(rim1y / L[1]);

        // get prior segment vectors
        lim2x = rim1x - rim2x;
        lim2y = rim1y - rim2y;

        lim1x = rix - rim1x;
        lim1y = riy - rim1y;

        // increment cell index
        ci++;
      }
    }
    // unwrapped vertex coordinate
    unwrappedX = cx + rix;
    unwrappedY = cy + riy;

    // preferred segment length
    l0i = l0[gi];

    // get next adjacent vertices
    rip1x = x[NDIM * ip1[gi]] - cx;
    rip1y = x[NDIM * ip1[gi] + 1] - cy;
    if (pbc[0])
      rip1x -= L[0] * round(rip1x / L[0]);
    if (pbc[1])
      rip1y -= L[1] * round(rip1y / L[1]);

    // -- Area force (comes from a cross product)
    forceX = 0.5 * fa * (rim1y - rip1y);
    forceY = 0.5 * fa * (rip1x - rim1x);
    int cellIndex, vertexIndex;
    cindices(cellIndex, vertexIndex, gi);
    F[NDIM * gi] += forceX;
    F[NDIM * gi + 1] += forceY;

    fieldShapeStress[gi][0] += unwrappedX * forceX;
    fieldShapeStress[gi][1] += unwrappedY * forceY;
    fieldShapeStress[gi][2] += unwrappedX * forceY;

    // -- Perimeter force
    lix = rip1x - rix;
    liy = rip1y - riy;

    // segment lengths
    lim1 = sqrt(lim1x * lim1x + lim1y * lim1y);
    li = sqrt(lix * lix + liy * liy);

    // segment deviations (note: m is prior vertex, p is next vertex i.e. gi - 1, gi + 1 mod the right number of vertices)
    dlim1 = (lim1 / l0im1) - 1.0;
    dli = (li / l0i) - 1.0;

    // segment forces
    flim1 = kl * (rho0 / l0im1);
    fli = kl * (rho0 / l0i);

    // add to forces
    forceX = (fli * dli * lix / li) - (flim1 * dlim1 * lim1x / lim1);
    forceY = (fli * dli * liy / li) - (flim1 * dlim1 * lim1y / lim1);
    F[NDIM * gi] += forceX;
    F[NDIM * gi + 1] += forceY;

    // note - Andrew here, confirmed that the shape stress matrix is diagonal as written
    fieldShapeStress[gi][0] += unwrappedX * forceX;
    fieldShapeStress[gi][1] += unwrappedY * forceY;
    fieldShapeStress[gi][2] += unwrappedX * forceY;

    // update potential energy
    U += 0.5 * kl * (dli * dli);
    cellU[ci] += 0.5 * kl * (dli * dli);

    // -- Bending force
    if (kb > 0) {
      // get ip2 for third angle
      rip2x = x[NDIM * ip1[ip1[gi]]] - cx;
      rip2y = x[NDIM * ip1[ip1[gi]] + 1] - cy;
      if (pbc[0])
        rip2x -= L[0] * round(rip2x / L[0]);
      if (pbc[1])
        rip2y -= L[1] * round(rip2y / L[1]);

      // get last segment length
      lip1x = rip2x - rip1x;
      lip1y = rip2y - rip1y;

      // get angles
      sinim1 = lim1x * lim2y - lim1y * lim2x;
      cosim1 = lim1x * lim2x + lim1y * lim2y;

      sini = lix * lim1y - liy * lim1x;
      cosi = lix * lim1x + liy * lim1y;

      sinip1 = lip1x * liy - lip1y * lix;
      cosip1 = lip1x * lix + lip1y * liy;

      // get normal vectors
      nim1x = lim1y;
      nim1y = -lim1x;

      nix = liy;
      niy = -lix;

      // get change in angles
      dtim1 = atan2(sinim1, cosim1) - t0[im1[gi]];
      dti = atan2(sini, cosi) - t0[gi];
      dtip1 = atan2(sinip1, cosip1) - t0[ip1[gi]];

      // get delta delta theta's
      ddtim1 = (dti - dtim1) / (lim1 * lim1);
      ddti = (dti - dtip1) / (li * li);

      // add to force
      F[NDIM * gi] += fb * (ddtim1 * nim1x + ddti * nix);
      F[NDIM * gi + 1] += fb * (ddtim1 * nim1y + ddti * niy);

      // update potential energy
      U += 0.5 * kb * (dti * dti);
      cellU[ci] += 0.5 * kb * (dti * dti);
    }

    // update old coordinates
    rim2x = rim1x;
    rim1x = rix;
    rix = rip1x;

    rim2y = rim1y;
    rim1y = riy;
    riy = rip1y;

    // update old segment vectors
    lim2x = lim1x;
    lim2y = lim1y;

    lim1x = lix;
    lim1y = liy;

    // update old preferred segment length
    l0im1 = l0i;
  }

  // normalize per-cell stress by preferred cell area
  for (int ci = 0; ci < NCELLS; ci++) {
    for (int vi = 0; vi < nv[ci]; vi++) {
      int gi = gindex(ci, vi);
      fieldShapeStressCells[ci][0] += fieldShapeStress[gi][0];
      fieldShapeStressCells[ci][1] += fieldShapeStress[gi][1];
      fieldShapeStressCells[ci][2] += fieldShapeStress[gi][2];
    }
    // nondimensionalize the stress
    fieldShapeStressCells[ci][0] *= rho0 / a0[ci];
    fieldShapeStressCells[ci][1] *= rho0 / a0[ci];
    fieldShapeStressCells[ci][2] *= rho0 / a0[ci];
  }
}

void dpm::vertexRepulsiveForces2D() {
  // local variables
  int ci, cj, gi, gj, vi, vj, bi, bj, pi, pj, boxid, sbtmp;
  double sij, rij, dx, dy, rho0;
  double ftmp, fx, fy;

  // sort particles
  sortNeighborLinkedList2D();

  // get fundamental length
  rho0 = sqrt(a0.at(0));

  // reset contact network
  fill(cij.begin(), cij.end(), 0);

  // loop over boxes in neighbor linked list
  for (bi = 0; bi < NBX; bi++) {
    // get start of list of vertices
    pi = head[bi];

    // loop over linked list
    while (pi > 0) {
      // real particle index
      gi = pi - 1;

      // next particle in list
      pj = list[pi];

      // loop down neighbors of pi in same cell
      while (pj > 0) {
        // real index of pj
        gj = pj - 1;

        if (gj == ip1[gi] || gj == im1[gi]) {
          pj = list[pj];
          continue;
        }

        // contact distance
        sij = r[gi] + r[gj];

        // particle distance
        dx = x[NDIM * gj] - x[NDIM * gi];
        if (pbc[0])
          dx -= L[0] * round(dx / L[0]);
        if (dx < sij) {
          dy = x[NDIM * gj + 1] - x[NDIM * gi + 1];
          if (pbc[1])
            dy -= L[1] * round(dy / L[1]);
          if (dy < sij) {
            rij = sqrt(dx * dx + dy * dy);
            if (rij < sij) {
              // force scale
              ftmp = kc * (1 - (rij / sij)) * (rho0 / sij);
              fx = ftmp * (dx / rij);
              fy = ftmp * (dy / rij);

              // add to forces
              F[NDIM * gi] -= fx;
              F[NDIM * gi + 1] -= fy;

              F[NDIM * gj] += fx;
              F[NDIM * gj + 1] += fy;

              // increase potential energy
              U += 0.5 * kc * pow((1 - (rij / sij)), 2.0);

              // add to virial stress
              stress[0] += dx * fx;
              stress[1] += dy * fy;
              stress[2] += 0.5 * (dx * fy + dy * fx);

              fieldStress[gi][0] += dx * fx;
              fieldStress[gi][1] += dy * fy;
              fieldStress[gi][2] += 0.5 * (dx * fy + dy * fx);

              // add to contacts
              cindices(ci, vi, gi);
              cindices(cj, vj, gj);
              cellU[ci] += 0.5 * kc * pow((1 - (rij / sij)), 2.0) / 2.0;
              cellU[cj] += 0.5 * kc * pow((1 - (rij / sij)), 2.0) / 2.0;

              if (ci > cj)
                cij[NCELLS * cj + ci - (cj + 1) * (cj + 2) / 2]++;
              else if (ci < cj)
                cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2]++;
            }
          }
        }

        // update pj
        pj = list[pj];
      }

      // test overlaps with forward neighboring cells
      for (bj = 0; bj < NNN; bj++) {
        // only check if boundaries permit
        if (nn[bi][bj] == -1)
          continue;

        // get first particle in neighboring cell
        pj = head[nn[bi][bj]];

        // loop down neighbors of pi in same cell
        while (pj > 0) {
          // real index of pj
          gj = pj - 1;

          if (gj == ip1[gi] || gj == im1[gi]) {
            pj = list[pj];
            continue;
          }
          // contact distance
          sij = r[gi] + r[gj];

          // particle distance
          dx = x[NDIM * gj] - x[NDIM * gi];
          if (pbc[0])
            dx -= L[0] * round(dx / L[0]);
          if (dx < sij) {
            dy = x[NDIM * gj + 1] - x[NDIM * gi + 1];
            if (pbc[1])
              dy -= L[1] * round(dy / L[1]);
            if (dy < sij) {
              rij = sqrt(dx * dx + dy * dy);
              if (rij < sij) {
                // force scale
                ftmp = kc * (1 - (rij / sij)) * (rho0 / sij);
                fx = ftmp * (dx / rij);
                fy = ftmp * (dy / rij);

                // add to forces
                F[NDIM * gi] -= fx;
                F[NDIM * gi + 1] -= fy;

                F[NDIM * gj] += fx;
                F[NDIM * gj + 1] += fy;

                // increase potential energy
                U += 0.5 * kc * pow((1 - (rij / sij)), 2.0);

                // add to virial stress
                stress[0] += dx * fx;
                stress[1] += dy * fy;
                stress[2] += 0.5 * (dx * fy + dy * fx);

                fieldStress[gi][0] += dx * fx;
                fieldStress[gi][1] += dy * fy;
                fieldStress[gi][2] += 0.5 * (dx * fy + dy * fx);

                // add to contacts
                cindices(ci, vi, gi);
                cindices(cj, vj, gj);

                cellU[ci] += 0.5 * kc * pow((1 - (rij / sij)), 2.0) / 2.0;
                cellU[cj] += 0.5 * kc * pow((1 - (rij / sij)), 2.0) / 2.0;

                if (ci > cj)
                  cij[NCELLS * cj + ci - (cj + 1) * (cj + 2) / 2]++;
                else if (ci < cj)
                  cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2]++;
              }
            }
          }

          // update pj
          pj = list[pj];
        }
      }

      // update pi index to be next
      pi = list[pi];
    }
  }

  // normalize stress by box area, make dimensionless
  // units explanation (Andrew): pressure is force/area. up til now, stress has been force times distance.
  // so after dividing by area, we have force / distance. We need to multiply one last time by rho0, done.

  stress[0] *= (rho0 / (L[0] * L[1]));
  stress[1] *= (rho0 / (L[0] * L[1]));
  stress[2] *= (rho0 / (L[0] * L[1]));

  // normalize per-cell stress by preferred cell area
  for (int ci = 0; ci < NCELLS; ci++) {
    for (int vi = 0; vi < nv[ci]; vi++) {
      int gi = gindex(ci, vi);
      fieldStressCells[ci][0] += fieldStress[gi][0];
      fieldStressCells[ci][1] += fieldStress[gi][1];
      fieldStressCells[ci][2] += fieldStress[gi][2];
    }
    fieldStressCells[ci][0] *= rho0 / a0[ci];
    fieldStressCells[ci][1] *= rho0 / a0[ci];
    fieldStressCells[ci][2] *= rho0 / a0[ci];
  }
}

void dpm::vertexAttractiveForces2D() {
  // local variables
  int ci, cj, gi, gj, vi, vj, bi, bj, pi, pj, boxid, sbtmp;
  double sij, rij, dx, dy, rho0;
  double ftmp, fx, fy;

  // attraction shell parameters
  double shellij, cutij, xij, kint = (kc * l1) / (l2 - l1);

  // sort particles
  sortNeighborLinkedList2D();

  // get fundamental length
  rho0 = sqrt(a0[0]);

  // reset contact network
  fill(cij.begin(), cij.end(), 0);

  // loop over boxes in neighbor linked list
  for (bi = 0; bi < NBX; bi++) {
    // get start of list of vertices
    pi = head[bi];

    // loop over linked list
    while (pi > 0) {
      // real particle index
      gi = pi - 1;

      // cell index of gi
      cindices(ci, vi, gi);

      // next particle in list
      pj = list[pi];

      // loop down neighbors of pi in same cell
      while (pj > 0) {
        // real index of pj
        gj = pj - 1;

        cindices(cj, vj, gj);

        if (gj == ip1[gi] || gj == im1[gi]) {
          pj = list[pj];
          continue;
        }

        // contact distance
        sij = r[gi] + r[gj];

        // attraction distances
        shellij = (1.0 + l2) * sij;
        cutij = (1.0 + l1) * sij;

        // particle distance
        dx = x[NDIM * gj] - x[NDIM * gi];
        if (pbc[0])
          dx -= L[0] * round(dx / L[0]);
        if (dx < shellij) {
          dy = x[NDIM * gj + 1] - x[NDIM * gi + 1];
          if (pbc[1])
            dy -= L[1] * round(dy / L[1]);
          if (dy < shellij) {
            rij = sqrt(dx * dx + dy * dy);
            if (rij < shellij) {
              // scaled distance
              xij = rij / sij;

              // pick force based on vertex-vertex distance
              if (rij > cutij) {
                // force scale
                ftmp = kint * (xij - 1.0 - l2) / sij;

                // increase potential energy
                U += -0.5 * kint * pow(1.0 + l2 - xij, 2.0);
                cellU[ci] += -0.5 * kint * pow(1.0 + l2 - xij, 2.0) / 2.0;
                cellU[cj] += -0.5 * kint * pow(1.0 + l2 - xij, 2.0) / 2.0;
              } else {
                // force scale
                ftmp = kc * (1 - xij) / sij;

                // increase potential energy
                U += 0.5 * kc * (pow(1.0 - xij, 2.0) - l1 * l2);
                cellU[ci] += 0.5 * kc * (pow(1.0 - xij, 2.0) - l1 * l2) / 2.0;
                cellU[cj] += 0.5 * kc * (pow(1.0 - xij, 2.0) - l1 * l2) / 2.0;
              }

              // force elements
              fx = ftmp * (dx / rij);
              fy = ftmp * (dy / rij);

              // add to forces
              F[NDIM * gi] -= fx;
              F[NDIM * gi + 1] -= fy;

              F[NDIM * gj] += fx;
              F[NDIM * gj + 1] += fy;

              // add to virial stress
              stress[0] += dx * fx;
              stress[1] += dy * fy;
              stress[2] += 0.5 * (dx * fy + dy * fx);

              fieldStress[gi][0] += dx * fx;
              fieldStress[gi][1] += dy * fy;
              fieldStress[gi][2] += 0.5 * (dx * fy + dy * fx);

              // cindices(cj, vj, gj);
              //  add to contacts
              if (ci > cj)
                cij[NCELLS * cj + ci - (cj + 1) * (cj + 2) / 2]++;
              else if (ci < cj)
                cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2]++;
            }
          }
        }

        // update pj
        pj = list[pj];
      }

      // test overlaps with forward neighboring cells
      for (bj = 0; bj < NNN; bj++) {
        // only check if boundaries permit
        if (nn[bi][bj] == -1)
          continue;

        // get first particle in neighboring cell
        pj = head[nn[bi][bj]];

        // loop down neighbors of pi in same cell
        while (pj > 0) {
          // real index of pj
          gj = pj - 1;

          cindices(cj, vj, gj);

          if (gj == ip1[gi] || gj == im1[gi]) {
            pj = list[pj];
            continue;
          }

          // contact distance
          sij = r[gi] + r[gj];

          // attraction distances
          shellij = (1.0 + l2) * sij;
          cutij = (1.0 + l1) * sij;

          // particle distance
          dx = x[NDIM * gj] - x[NDIM * gi];
          if (pbc[0])
            dx -= L[0] * round(dx / L[0]);
          if (dx < shellij) {
            dy = x[NDIM * gj + 1] - x[NDIM * gi + 1];
            if (pbc[1])
              dy -= L[1] * round(dy / L[1]);
            if (dy < shellij) {
              rij = sqrt(dx * dx + dy * dy);
              if (rij < shellij) {
                // scaled distance
                xij = rij / sij;

                // pick force based on vertex-vertex distance
                if (rij > cutij) {
                  // force scale
                  ftmp = kint * (xij - 1.0 - l2) / sij;

                  // increase potential energy
                  U += -0.5 * kint * pow(1.0 + l2 - xij, 2.0);
                  cellU[ci] += -0.5 * kint * pow(1.0 + l2 - xij, 2.0) / 2.0;
                  cellU[cj] += -0.5 * kint * pow(1.0 + l2 - xij, 2.0) / 2.0;
                } else {
                  // force scale
                  ftmp = kc * (1 - xij) / sij;

                  // increase potential energy
                  U += 0.5 * kc * (pow(1.0 - xij, 2.0) - l1 * l2);
                  cellU[ci] += 0.5 * kc * (pow(1.0 - xij, 2.0) - l1 * l2) / 2.0;
                  cellU[cj] += 0.5 * kc * (pow(1.0 - xij, 2.0) - l1 * l2) / 2.0;
                }

                // force elements
                fx = ftmp * (dx / rij);
                fy = ftmp * (dy / rij);

                // add to forces
                F[NDIM * gi] -= fx;
                F[NDIM * gi + 1] -= fy;

                F[NDIM * gj] += fx;
                F[NDIM * gj + 1] += fy;

                // add to virial stress
                stress[0] += dx * fx;
                stress[1] += dy * fy;
                stress[2] += 0.5 * (dx * fy + dy * fx);

                fieldStress[gi][0] += dx * fx;
                fieldStress[gi][1] += dy * fy;
                fieldStress[gi][2] += 0.5 * (dx * fy + dy * fx);

                if (ci > cj)
                  cij[NCELLS * cj + ci - (cj + 1) * (cj + 2) / 2]++;
                else if (ci < cj)
                  cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2]++;
              }
            }
          }

          // update pj
          pj = list[pj];
        }
      }

      // update pi index to be next
      pi = list[pi];
    }
  }

  // normalize stress by box area, make dimensionless
  stress[0] *= (rho0 / (L[0] * L[1]));
  stress[1] *= (rho0 / (L[0] * L[1]));
  stress[2] *= (rho0 / (L[0] * L[1]));

  // normalize per-cell stress by cell area
  for (int ci = 0; ci < NCELLS; ci++) {
    for (int vi = 0; vi < nv[ci]; vi++) {
      int gi = gindex(ci, vi);
      fieldStressCells[ci][0] += fieldStress[gi][0];
      fieldStressCells[ci][1] += fieldStress[gi][1];
      fieldStressCells[ci][2] += fieldStress[gi][2];
    }
    fieldStressCells[ci][0] *= rho0 / a0[ci];
    fieldStressCells[ci][1] *= rho0 / a0[ci];
    fieldStressCells[ci][2] *= rho0 / a0[ci];
  }
}
void dpm::repulsiveForceUpdate() {
  resetForcesAndEnergy();
  shapeForces2D();
  vertexRepulsiveForces2D();
}

void dpm::attractiveForceUpdate() {
  resetForcesAndEnergy();
  shapeForces2D();
  vertexAttractiveForces2D();
}

/******************************

        D P M

                I N T E G R A T O R S

*******************************/

void dpm::setdt(double dt0) {
  // local variables
  int i = 0;
  double ta = 0, tl = 0, tb = 0, tmin = 0, rho0 = 0;

  // typical length
  rho0 = sqrt(a0.at(0));

  // set typical time scales
  ta = rho0 / sqrt(ka);
  tl = (rho0 * l0.at(0)) / sqrt(ka * kl);
  tb = (rho0 * l0.at(0)) / sqrt(ka * kb);

  // set main time scale as min
  tmin = 1e8;
  if (ta < tmin)
    tmin = ta;
  if (tl < tmin)
    tmin = tl;
  if (tb < tmin)
    tmin = tb;

  // set dt
  dt = dt0 * tmin;
}

void dpm::vertexFIRE2D(dpmMemFn forceCall, double Ftol, double dt0) {
  // local variables
  int i;
  double rho0;

  // check to see if cell linked-list has been initialized
  if (NBX == -1) {
    cerr << "	** ERROR: In dpm::fire, NBX = -1, so cell linked-list has not yet been initialized. Ending here.\n";
    exit(1);
  }

  // FIRE variables
  double P, fnorm, fcheck, vnorm, alpha, dtmax, dtmin;
  int npPos, npNeg, fireit;

  // set dt based on geometric parameters
  setdt(dt0);

  // Initialize FIRE variables
  P = 0;
  fnorm = 0;
  vnorm = 0;
  alpha = alpha0;

  dtmax = 10.0 * dt;
  dtmin = 1e-2 * dt;

  npPos = 0;
  npNeg = 0;

  fireit = 0;
  fcheck = 10 * Ftol;

  // reset forces and velocities
  resetForcesAndEnergy();
  fill(v.begin(), v.end(), 0.0);

  // length scale
  rho0 = sqrt(a0.at(0));

  // relax forces using FIRE
  while (fcheck > Ftol && fireit < itmax) {
    // compute P
    P = 0.0;
    for (i = 0; i < vertDOF; i++)
      P += v[i] * F[i];

    // print to console
    if (fireit % NSKIP == 0) {
      cout << endl
           << endl;
      cout << "===========================================" << endl;
      cout << " 	F I R E 						" << endl;
      cout << "		M I N I M I Z A T I O N 	" << endl;
      cout << "===========================================" << endl;
      cout << endl;
      cout << "	** fireit 	= " << fireit << endl;
      cout << "	** fcheck 	= " << fcheck << endl;
      cout << "	** U 		= " << U << endl;
      cout << "	** dt 		= " << dt << endl;
      cout << "	** P 		= " << P << endl;
      cout << " ** phi (square boundaries)  = " << vertexPreferredPackingFraction2D() << endl;
      cout << "	** alpha 	= " << alpha << endl;
      cout << "	** npPos 	= " << npPos << endl;
      cout << "	** npNeg 	= " << npNeg << endl;
      cout << "	** sxx  	= " << stress[0] << endl;
      cout << "	** syy 		= " << stress[1] << endl;
      cout << "	** sxy 		= " << stress[2] << endl;
    }

    // Adjust simulation based on net motion of degrees of freedom
    if (P > 0) {
      // increase positive counter
      npPos++;

      // reset negative counter
      npNeg = 0;

      // alter simulation if enough positive steps have been taken
      if (npPos > NDELAY) {
        // change time step
        if (dt * finc < dtmax)
          dt *= finc;

        // decrease alpha
        alpha *= falpha;
      }
    } else {
      // reset positive counter
      npPos = 0;

      // increase negative counter
      npNeg++;

      // check if simulation is stuck
      if (npNeg > NNEGMAX) {
        cerr << "	** ERROR: During initial FIRE minimization, P < 0 for too long, so ending." << endl;
        exit(1);
      }

      // take half step backwards, reset velocities
      for (i = 0; i < vertDOF; i++) {
        // take half step backwards
        x[i] -= 0.5 * dt * v[i];

        // reset vertex velocities
        v[i] = 0.0;
      }

      // decrease time step if past initial delay
      if (fireit > NDELAY) {
        // decrease time step
        if (dt * fdec > dtmin)
          dt *= fdec;

        // reset alpha
        alpha = alpha0;
      }
    }
    // VV VELOCITY UPDATE #1
    for (i = 0; i < vertDOF; i++)
      v[i] += 0.5 * dt * F[i];

    // compute fnorm, vnorm and P
    fnorm = 0.0;
    vnorm = 0.0;
    for (i = 0; i < vertDOF; i++) {
      fnorm += F[i] * F[i];
      vnorm += v[i] * v[i];
    }
    fnorm = sqrt(fnorm);
    vnorm = sqrt(vnorm);

    // update velocities (s.d. vs inertial dynamics) only if forces are acting
    if (fnorm > 0) {
      for (i = 0; i < vertDOF; i++)
        v[i] = (1 - alpha) * v[i] + alpha * (F[i] / fnorm) * vnorm;
    }
    // VV POSITION UPDATE
    for (i = 0; i < vertDOF; i++) {
      // update position
      x[i] += dt * v[i];

      // recenter in box
      if (x[i] > L[i % NDIM] && pbc[i % NDIM])
        x[i] -= L[i % NDIM];
      else if (x[i] < 0 && pbc[i % NDIM])
        x[i] += L[i % NDIM];
    }

    // update forces (function passed as argument)
    CALL_MEMBER_FN(*this, forceCall)
    ();

    /*cout << "vertex fire after force call \n\n\n\n";
    for (int k = 0; k < F.size(); k += 2) {
      cout << "Fx,Fy = " << F[k * NDIM] << '\t' << F[k * NDIM + 1] << '\n';
      cout << "rx, ry = " << x[k * NDIM] << '\t' << x[k * NDIM + 1] << '\n';
    }*/

    // VV VELOCITY UPDATE #2
    for (i = 0; i < vertDOF; i++)
      v[i] += 0.5 * F[i] * dt;

    // update fcheck based on fnorm (= force per degree of freedom)
    fcheck = 0.0;
    for (i = 0; i < vertDOF; i++)
      fcheck += F[i] * F[i];
    fcheck = sqrt(fcheck / vertDOF);

    // update iterator
    fireit++;
  }
  // check if FIRE converged
  if (fireit == itmax) {
    cout << "	** FIRE minimization did not converge, fireit = " << fireit << ", itmax = " << itmax << "; ending." << endl;
    exit(1);
  } else {
    cout << endl;
    cout << "===========================================" << endl;
    cout << " 	F I R E 						" << endl;
    cout << "		M I N I M I Z A T I O N 	" << endl;
    cout << "	C O N V E R G E D! 				" << endl;
    cout << "===========================================" << endl;
    cout << endl;
    cout << "	** fireit 	= " << fireit << endl;
    cout << "	** fcheck 	= " << fcheck << endl;
    cout << "	** U 		= " << U << endl;

    cout << "	** fnorm	= " << fnorm << endl;
    cout << "	** vnorm 	= " << vnorm << endl;
    cout << "	** dt 		= " << dt << endl;
    cout << "	** P 		= " << P << endl;
    cout << "	** alpha 	= " << alpha << endl;
    cout << "	** sxx  	= " << stress[0] << endl;
    cout << "	** syy 		= " << stress[1] << endl;
    cout << "	** sxy 		= " << stress[2] << endl;
    cout << endl
         << endl;
  }
}

void dpm::setBlockGridDims(int dimBlock) {
  // the argument determined the block dimension, we're going to use a 1D indexing for our kernel
  // gridDim is going to be approximately number of total vertices divided by the size of dimBlock
  //  with some additional algebra done to account for when NVTOT is not evenly divided by dimBlock
  dimGrid = (NVTOT + dimBlock - 1) / dimBlock;
  cout << "setting blockDim = " << dimBlock << '\n';
  cout << "setting gridDim = " << dimGrid << '\n';
}

void dpm::setDeviceVariables(double boxlengthX, double boxlengthY, double density) {
  cudaSetDevice(0);
  // set device variables needed for force kernel
  cudaError_t cudaStatus;

  int temp_NVTOT = NVTOT;
  double temp_L[2];
  temp_L[0] = boxlengthX;
  temp_L[1] = boxlengthY;
  double temp_rho0 = density;
  double temp_kc = kc;

  cout << "NVTOT = " << temp_NVTOT << ", L[0] = " << temp_L[0] << ", kc = " << temp_kc << ", rho0 = " << temp_rho0 << '\n';
  // printf("before setting device variables: d_numVertices = %d, d_L[0] = %f, d_kc = %f, d_rho0 = %f\n", d_numVertices, d_L[0], d_kc, d_rho0);

  printf("number of bytes to copy: %d %d %d %d \n", sizeof(int32_t), 2 * sizeof(double), sizeof(temp_rho0), sizeof(temp_kc));

  cudaStatus = cudaMemcpyToSymbol(d_numVertsPerCell, &nv[0], sizeof(int));
  if (cudaStatus != cudaSuccess) {
    cout << "error: failed to read in nv[0]\n";
    cout << cudaGetErrorString(cudaStatus) << '\n';
  }

  cudaStatus = cudaMemcpyToSymbol(d_numVertices, &temp_NVTOT, sizeof(int));
  if (cudaStatus != cudaSuccess) {
    cout << "error: failed to read in NVTOT\n";
    cout << cudaGetErrorString(cudaStatus) << '\n';
  }

  cudaStatus = cudaMemcpyToSymbol(d_L, temp_L, 2 * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    cout << "error: failed to read in L\n";
    cout << cudaGetErrorString(cudaStatus) << '\n';
  }

  cudaStatus = cudaMemcpyToSymbol(d_rho0, &temp_rho0, sizeof(double));
  if (cudaStatus != cudaSuccess) {
    cout << "error: failed to read in rho0\n";
    cout << cudaGetErrorString(cudaStatus) << '\n';
  }

  cudaStatus = cudaMemcpyToSymbol(d_kc, &temp_kc, sizeof(double));
  if (cudaStatus != cudaSuccess) {
    cout << "error: failed to read in kc\n";
    cout << cudaGetErrorString(cudaStatus) << '\n';
  }

  // printf("after setting device variables: d_numVertices = %d, d_L[0] = %f, d_kc = %f, d_rho0 = %f\n", d_numVertices, d_L[0], d_kc, d_rho0);
}

void dpm::cudaVertexNVE(ofstream& enout, double T, double dt0, int NT, int NPRINTSKIP) {
  // strategy: looks mostly like vertexNVE2D, but uses a CUDA kernel to compute forces
  // local variables
  int t, i;
  double K, simclock;
  int dimBlock = 1024;
  cudaEvent_t start, stop;  // using cuda events to measure time
  float elapsed_time_ms;    // which is applicable for asynchronous code also

  // calls to set cuda-related variables
  setBlockGridDims(dimBlock);
  setDeviceVariables(L[0], L[1], sqrt(a0[0]));

  // set time step magnitude
  setdt(dt0);

  // initialize time keeper
  simclock = 0.0;

  // initialize velocities
  drawVelocities2D(T);

  size_t sizeR = r.size() * sizeof(double);
  size_t sizeX = x.size() * sizeof(double);
  size_t sizeF = F.size() * sizeof(double);
  size_t sizeVertexEnergy = vertexEnergy.size() * sizeof(double);

  cudaMalloc((void**)&dev_r, sizeR);  // allocate memory on device
  cudaMalloc((void**)&dev_x, sizeX);
  cudaMalloc((void**)&dev_F, sizeF);
  cudaMalloc((void**)&dev_vertexEnergy, sizeVertexEnergy);

  double *dev_r, *dev_x, *dev_F, *dev_vertexEnergy;

  // loop over time, print energy
  for (t = 0; t < NT; t++) {
    // VV VELOCITY UPDATE #1
    for (i = 0; i < vertDOF; i++)
      v[i] += 0.5 * dt * F[i];

    // VV POSITION UPDATE
    for (i = 0; i < vertDOF; i++) {
      // update position
      x[i] += dt * v[i];

      // recenter in box
      if (x[i] > L[i % NDIM] && pbc[i % NDIM])
        x[i] -= L[i % NDIM];
      else if (x[i] < 0 && pbc[i % NDIM])
        x[i] += L[i % NDIM];
    }

    // kernelVertexForces has input arrays: rad, pos, force array (empty)
    // kernelVertexForces has output : force array (full), energy
    // setDeviceVariables();

    printf("Launching kernel\n");

    cudaEventCreate(&start);  // instrument code to measure start time
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // these are done serially (for now), can be made parallel in future work
    // timed for fairness in comparing speeds with serial code
    resetForcesAndEnergy();
    shapeForces2D();

    cudaMemcpy(dev_r, &r[0], sizeR, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, &x[0], sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_F, &F[0], sizeF, cudaMemcpyHostToDevice);

    // FORCE UPDATE
    kernelVertexForces<<<dimGrid, dimBlock>>>(dev_r, dev_x, dev_F, dev_vertexEnergy);

    cudaEventRecord(stop, 0);  // instrument code to measure end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    printf("Back from kernel\n");
    cudaMemcpy(&F[0], dev_F, sizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(&vertexEnergy[0], dev_vertexEnergy, sizeVertexEnergy, cudaMemcpyDeviceToHost);

    printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);  // exec. time

    for (i = 0; i < NVTOT; i++) {
      U += vertexEnergy[i];
    }

    // VV VELOCITY UPDATE #2
    for (i = 0; i < vertDOF; i++)
      v[i] += 0.5 * F[i] * dt;

    // update sim clock
    simclock += dt;

    // print to console and file
    if (t % NPRINTSKIP == 0) {
      // compute kinetic energy
      K = vertexKineticEnergy();

      // print to console
      cout << endl
           << endl;
      cout << "===============================" << endl;
      cout << "	D P M  						" << endl;
      cout << " 			 					" << endl;
      cout << "		N V E 					" << endl;
      cout << "===============================" << endl;
      cout << endl;
      cout << "	** t / NT	= " << t << " / " << NT << endl;
      cout << "	** U 		= " << setprecision(12) << U << endl;
      cout << "	** K 		= " << setprecision(12) << K << endl;
      cout << "	** E 		= " << setprecision(12) << U + K << endl;

      // print to energy file
      cout << "** printing energy" << endl;
      enout << setw(w) << left << t;
      enout << setw(wnum) << left << simclock;
      enout << setw(wnum) << setprecision(12) << U;
      enout << setw(wnum) << setprecision(12) << K;
      enout << setw(wnum) << setprecision(12) << U + K;
      enout << endl;

      // print to configuration only if position file is open
      if (posout.is_open())
        printConfiguration2D();
    }
  }
}

void dpm::vertexNVE2D(ofstream& enout, dpmMemFn forceCall, double T, double dt0, int NT, int NPRINTSKIP) {
  // local variables
  int t, i;
  double K, simclock;

  // set time step magnitude
  setdt(dt0);

  // initialize time keeper
  simclock = 0.0;

  // initialize velocities
  drawVelocities2D(T);

  // loop over time, print energy
  for (t = 0; t < NT; t++) {
    // VV VELOCITY UPDATE #1
    for (i = 0; i < vertDOF; i++)
      v[i] += 0.5 * dt * F[i];

    // VV POSITION UPDATE
    for (i = 0; i < vertDOF; i++) {
      // update position
      x[i] += dt * v[i];

      // recenter in box
      if (x[i] > L[i % NDIM] && pbc[i % NDIM])
        x[i] -= L[i % NDIM];
      else if (x[i] < 0 && pbc[i % NDIM])
        x[i] += L[i % NDIM];
    }

    // FORCE UPDATE
    CALL_MEMBER_FN(*this, forceCall)
    ();

    // VV VELOCITY UPDATE #2
    for (i = 0; i < vertDOF; i++)
      v[i] += 0.5 * F[i] * dt;

    // update sim clock
    simclock += dt;

    // print to console and file
    if (t % NPRINTSKIP == 0) {
      // compute kinetic energy
      K = vertexKineticEnergy();

      // print to console
      cout << endl
           << endl;
      cout << "===============================" << endl;
      cout << "	D P M  						" << endl;
      cout << " 			 					" << endl;
      cout << "		N V E 					" << endl;
      cout << "===============================" << endl;
      cout << endl;
      cout << "	** t / NT	= " << t << " / " << NT << endl;
      cout << "	** U 		= " << setprecision(12) << U << endl;
      cout << "	** K 		= " << setprecision(12) << K << endl;
      cout << "	** E 		= " << setprecision(12) << U + K << endl;

      // print to energy file
      cout << "** printing energy" << endl;
      enout << setw(w) << left << t;
      enout << setw(wnum) << left << simclock;
      enout << setw(wnum) << setprecision(12) << U;
      enout << setw(wnum) << setprecision(12) << K;
      enout << setw(wnum) << setprecision(12) << U + K;
      enout << endl;

      // print to configuration only if position file is open
      if (posout.is_open())
        printConfiguration2D();
    }
  }
}

/******************************

        D P M

                P R O T O C O L S

*******************************/

void dpm::vertexCompress2Target2D(dpmMemFn forceCall, double Ftol, double dt0, double phi0Target, double dphi0) {
  // local variables
  int it = 0, itmax = 1e4;
  double phi0 = vertexPreferredPackingFraction2D();
  double scaleFactor = 1.0, P, Sxy;

  // loop while phi0 < phi0Target
  while (phi0 < phi0Target && it < itmax) {
    // scale particle sizes
    scaleParticleSizes2D(scaleFactor);

    // update phi0
    phi0 = vertexPreferredPackingFraction2D();

    // relax configuration (pass member function force update)
    vertexFIRE2D(forceCall, Ftol, dt0);

    // get scale factor
    scaleFactor = sqrt((phi0 + dphi0) / phi0);

    // get updated pressure
    P = 0.5 * (stress[0] + stress[1]);
    Sxy = stress[2];

    // print to console
    if (it % 50 == 0) {
      cout << endl
           << endl;
      cout << "===============================" << endl;
      cout << "								" << endl;
      cout << " 	C O M P R E S S I O N 		" << endl;
      cout << "								" << endl;
      cout << "	P R O T O C O L 	  		" << endl;
      cout << "								" << endl;
      cout << "===============================" << endl;
      cout << endl;
      cout << "	** it 			= " << it << endl;
      cout << "	** phi0 curr	= " << phi0 << endl;
      if (phi0 + dphi0 < phi0Target)
        cout << "	** phi0 next 	= " << phi0 + dphi0 << endl;
      cout << "	** P 			= " << P << endl;
      cout << "	** Sxy 			= " << Sxy << endl;
      cout << "	** U 			= " << U << endl;
      // printConfiguration2D();
      cout << endl
           << endl;

      // update iterate
      it++;
    }
  }
}
void dpm::vertexJamming2D(dpmMemFn forceCall, double Ftol, double Ptol, double dt0, double dphi0, bool plotCompression) {
  // local variables
  int k = 0, nr;
  bool jammed = 0, overcompressed = 0, undercompressed = 0;
  double pcheck, phi0, rH, r0, rL, rho0, scaleFactor = 1.0;
  // double pcheck, phi0, rH, r0, rL, rho0, scaleFactor;

  // initialize binary root search parameters
  r0 = sqrt(a0.at(0));
  rH = -1;
  rL = -1;

  // initialize preferred packing fraction
  phi0 = vertexPreferredPackingFraction2D();

  // save initial state
  vector<double> xsave(vertDOF, 0.0);
  vector<double> rsave(vertDOF, 0.0);
  vector<double> l0save(vertDOF, 0.0);
  vector<double> t0save(vertDOF, 0.0);
  vector<double> a0save(vertDOF, 0.0);

  xsave = x;
  rsave = r;
  l0save = l0;
  t0save = t0;
  a0save = a0;

  // loop until jamming is found
  while (!jammed && k < itmax) {
    // set length scale by 1st particle preferred area
    rho0 = sqrt(a0.at(0));

    // relax configuration (pass member function force update)
    vertexFIRE2D(forceCall, Ftol, dt0);

    // update pressure
    pcheck = 0.5 * (stress[0] + stress[1]);

    // remove rattlers
    nr = removeRattlers();

    // boolean checks for jamming
    undercompressed = ((pcheck < 2.0 * Ptol && rH < 0) || (pcheck < Ptol && rH > 0));
    overcompressed = (pcheck > 2.0 * Ptol);
    jammed = (pcheck < 2.0 * Ptol && pcheck > Ptol && rH > 0 && rL > 0);

    // output to console
    cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;
    cout << "===============================================" << endl
         << endl;
    cout << " 	Q U A S I S T A T I C  						" << endl;
    cout << " 	  	I S O T R O P I C 						" << endl;
    cout << "			C O M P R E S S I O N 				" << endl
         << endl;
    cout << "===============================================" << endl;
    cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;
    cout << endl;
    cout << "	* k 			= " << k << endl;
    cout << "	* phi0 			= " << phi0 << endl;
    cout << "	* phi 			= " << vertexPackingFraction2D() << endl;
    cout << "	* scaleFactor 	= " << scaleFactor << endl;
    cout << "	* r0 			= " << r0 << endl;
    cout << "	* rH 			= " << rH << endl;
    cout << "	* rL 			= " << rL << endl;
    cout << "	* pcheck 		= " << pcheck << endl;
    cout << "	* U 		 	= " << U << endl;
    cout << "	* Nvv  			= " << vvContacts() << endl;
    cout << "	* Ncc 			= " << ccContacts() << endl;
    cout << "	* # of rattlers = " << nr << endl
         << endl;
    cout << "	* undercompressed = " << undercompressed << endl;
    cout << "	* overcompressed = " << overcompressed << endl;
    cout << "	* jammed = " << jammed << endl
         << endl;
    if (plotCompression)
      printConfiguration2D();
    cout << endl
         << endl;

    // update particle scaleFactor based on target check
    if (rH < 0) {
      // if still undercompressed, then grow until overcompressed found
      if (undercompressed) {
        r0 = rho0;
        scaleFactor = sqrt((phi0 + dphi0) / phi0);
      }
      // if first overcompressed, decompress by dphi/2 until unjamming
      else if (overcompressed) {
        // current = upper bound length scale r
        rH = rho0;

        // save first overcompressed state
        r0 = rH;
        xsave = x;
        rsave = r;
        l0save = l0;
        t0save = t0;
        a0save = a0;

        // shrink particle sizes
        scaleFactor = sqrt((phi0 - 0.5 * dphi0) / phi0);

        // print to console
        cout << "	-- -- overcompressed for the first time, scaleFactor = " << scaleFactor << endl;
      }
    } else {
      if (rL < 0) {
        // if first undercompressed, save last overcompressed state, begin root search
        if (undercompressed) {
          // current = new lower bound length scale r
          rL = rho0;

          // load state
          x = xsave;
          r = rsave;
          l0 = l0save;
          t0 = t0save;
          a0 = a0save;

          // compute new scale factor by root search
          scaleFactor = 0.5 * (rH + rL) / r0;

          // print to console
          cout << "	-- -- undercompressed for the first time, scaleFactor = " << scaleFactor << endl;
          cout << "	-- -- BEGINNING ROOT SEARCH IN ENTHALPY MIN PROTOCOL..." << endl;
        }
        // if still overcompressed, decrement again
        else if (overcompressed) {
          // current = upper bound length scale r
          rH = rho0;

          // save overcompressed state
          r0 = rH;
          xsave = x;
          rsave = r;
          l0save = l0;
          t0save = t0;
          a0save = a0;

          // keep shrinking at same rate until unjamming
          scaleFactor = sqrt((phi0 - 0.5 * dphi0) / phi0);

          // print to console
          cout << "	-- -- overcompressed, still no unjamming, scaleFactor = " << scaleFactor << endl;
        }
      } else {
        // if found undercompressed state, go to state between undercompressed and last overcompressed states (from saved state)
        if (undercompressed) {
          // current = new lower bound length scale r
          rL = rho0;

          // load state
          x = xsave;
          r = rsave;
          l0 = l0save;
          t0 = t0save;
          a0 = a0save;

          // compute new scale factor by root search
          scaleFactor = 0.5 * (rH + rL) / r0;

          // print to console
          cout << "	-- -- undercompressed, scaleFactor = " << scaleFactor << endl;
        } else if (overcompressed) {
          // current = upper bound length scale r
          rH = rho0;

          // load state
          x = xsave;
          r = rsave;
          l0 = l0save;
          t0 = t0save;
          a0 = a0save;

          // compute new scale factor
          scaleFactor = 0.5 * (rH + rL) / r0;

          // print to console
          cout << "	-- -- overcompressed, scaleFactor = " << scaleFactor << endl;
        } else if (jammed) {
          cout << "	** At k = " << k << ", target pressure found!" << endl;
          cout << " WRITING ENTHALPY-MINIMIZED CONFIG TO FILE" << endl;
          cout << " ENDING COMPRESSION SIMULATION" << endl;
          scaleFactor = 1.0;
          if (!plotCompression)
            printConfiguration2D();
          break;
        }
      }
    }

    // scale particle sizes
    scaleParticleSizes2D(scaleFactor);

    // update packing fraction
    phi0 = vertexPreferredPackingFraction2D();

    // update iterate
    k++;
  }
}

/******************************

        P R I N T   T O

        C O N S O L E  &  F I L E

*******************************/

void dpm::printContactMatrix() {
  int ci, cj;

  for (ci = 0; ci < NCELLS; ci++) {
    for (cj = 0; cj < NCELLS; cj++) {
      if (ci > cj)
        cout << setw(5) << cij[NCELLS * cj + ci - (cj + 1) * (cj + 2) / 2];
      else if (ci < cj)
        cout << setw(5) << cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2];
      else
        cout << setw(5) << 0;
    }
    cout << endl;
  }
}

void dpm::printConfiguration2D() {
  // local variables
  int ci, cj, vi, gi, ctmp, zc, zv;
  double xi, yi, dx, dy, Lx, Ly;

  // check if pos object is open
  if (!posout.is_open()) {
    cerr << "** ERROR: in printConfiguration2D, posout is not open, but function call will try to use. Ending here." << endl;
    exit(1);
  } else
    cout << "** In printConfiguration2D, printing particle positions to file..." << endl;

  // save box sizes
  Lx = L.at(0);
  Ly = L.at(1);

  // print information starting information
  posout << setw(w) << left << "NEWFR"
         << " " << endl;
  posout << setw(w) << left << "NUMCL" << setw(w) << left << NCELLS << endl;
  posout << setw(w) << left << "PACKF" << setw(wnum) << setprecision(pnum) << left << vertexPackingFraction2D() << endl;

  // print box sizes
  posout << setw(w) << left << "BOXSZ";
  posout << setw(wnum) << setprecision(pnum) << left << Lx;
  posout << setw(wnum) << setprecision(pnum) << left << Ly;
  posout << endl;

  // print stress info
  posout << setw(w) << left << "STRSS";
  posout << setw(wnum) << setprecision(pnum) << left << stress.at(0);
  posout << setw(wnum) << setprecision(pnum) << left << stress.at(1);
  posout << setw(wnum) << setprecision(pnum) << left << stress.at(2);
  posout << endl;

  // print coordinate for rest of the cells
  for (ci = 0; ci < NCELLS; ci++) {
    // get cell contact data
    zc = 0;
    zv = 0;
    for (cj = 0; cj < NCELLS; cj++) {
      if (ci != cj) {
        // contact info from entry ci, cj
        if (ci < cj)
          ctmp = cij[NCELLS * ci + cj - (ci + 1) * (ci + 2) / 2];
        else
          ctmp = cij[NCELLS * cj + ci - (cj + 1) * (cj + 2) / 2];

        // add to contact information
        zv += ctmp;
        if (ctmp > 0)
          zc++;
      }
    }

    // cell information
    posout << setw(w) << left << "CINFO";
    posout << setw(w) << left << nv.at(ci);
    posout << setw(w) << left << zc;
    posout << setw(w) << left << zv;
    posout << setw(wnum) << left << a0.at(ci);
    posout << setw(wnum) << left << area(ci);
    posout << setw(wnum) << left << perimeter(ci);
    posout << endl;

    // get initial vertex positions
    gi = gindex(ci, 0);
    xi = x.at(NDIM * gi);
    yi = x.at(NDIM * gi + 1);

    // place back in box center
    if (pbc[0])
      xi = fmod(xi, Lx);
    if (pbc[1])
      yi = fmod(yi, Ly);

    posout << setw(w) << left << "VINFO";
    posout << setw(w) << left << ci;
    posout << setw(w) << left << 0;

    // output initial vertex information
    posout << setw(wnum) << setprecision(pnum) << right << xi;
    posout << setw(wnum) << setprecision(pnum) << right << yi;
    posout << setw(wnum) << setprecision(pnum) << right << r.at(gi);
    posout << setw(wnum) << setprecision(pnum) << right << l0.at(gi);
    posout << setw(wnum) << setprecision(pnum) << right << t0.at(gi);
    posout << endl;

    // vertex information for next vertices
    for (vi = 1; vi < nv.at(ci); vi++) {
      // get global vertex index for next vertex
      gi++;

      // get next vertex positions
      dx = x.at(NDIM * gi) - xi;
      if (pbc[0])
        dx -= Lx * round(dx / Lx);
      xi += dx;

      dy = x.at(NDIM * gi + 1) - yi;
      if (pbc[1])
        dy -= Ly * round(dy / Ly);
      yi += dy;

      // Print indexing information
      posout << setw(w) << left << "VINFO";
      posout << setw(w) << left << ci;
      posout << setw(w) << left << vi;

      // output vertex information
      posout << setw(wnum) << setprecision(pnum) << right << xi;
      posout << setw(wnum) << setprecision(pnum) << right << yi;
      posout << setw(wnum) << setprecision(pnum) << right << r.at(gi);
      posout << setw(wnum) << setprecision(pnum) << right << l0.at(gi);
      posout << setw(wnum) << setprecision(pnum) << right << t0.at(gi);
      posout << endl;
    }
  }

  // print end frame
  posout << setw(w) << left << "ENDFR"
         << " " << endl;
}
