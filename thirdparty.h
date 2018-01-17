
/*
  This file is part of CycleSolver.

  CycleSolver is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  CycleSolver is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with CycleSolver. If not, see <http://www.gnu.org/licenses/>.
*/


namespace CycleSolver{


#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#define	MIN(A, B)	((A) < (B) ? (A) : (B))

#define EPS 1e-6
#define INF 1e30

  ///////////////////////////////////////////////////////////////
  // Fast min-sum
  // P. Felzenszwalb, J. McAuley
  // Fast Inference with Min-Sum Matrix Product
  // IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 33, No. 12, December 2011
  // Code from the second author's website
  ///////////////////////////////////////////////////////////////

  /** Find the row minima of A and the column minima of B */
  void minrowcol(std::vector<double> & A, std::vector<double> & B, int a, int b, int c, int d,
		 double* minrowA, double* mincolB) {
    int idx;
    for (int i = 0; i < a; i++){
      minrowA[i]=INF;
    }
    for(int j = 0; j < d; j++){
      mincolB[j]=INF;
    }

    idx=0;
    for (int j = 0; j < b; j++){
      for (int i = 0; i < a; i++){
	minrowA[i] = MIN(minrowA[i], A[idx]);
	idx++;
      }
    }

    idx=0;
    for (int k = 0; k < d; k++) {
      for (int j = 0; j < c; j++){
	mincolB[k] = MIN(mincolB[k], B[idx]);
	idx++;
      }
    }

  }

  /** Normalize A and B by subtracting their row and column minima, respectively */
  void normalize(std::vector<double> & A, std::vector<double> & B, int a, int b, int c, double* minrowA,
		 double* mincolB) {

    int idx;
    idx=0;
    for (int j = 0; j < b; j++)
      for (int i = 0; i < a; i++){
	A[idx] -= minrowA[i];  // idx = i+j*a
	idx++;
      }
    idx=0;
    for (int k = 0; k < c; k++)
      for (int j = 0; j < b; j++){
	B[idx] -= mincolB[k];  // idx = j + k*b
	idx++;
      }
  }

  /** Undo the normalization applied above */
  void unnormalize(std::vector<double> & A, std::vector<double> & B, std::vector<double> & C, int a, int b, int c,
		   double* minrowA, double* mincolB) {
    int idx;
    idx=0;
    for (int k = 0; k < c; k++)
      for (int i = 0; i < a; i++){
	C[idx] += minrowA[i] + mincolB[k];
	idx++;
      }
    idx=0;
    for (int j = 0; j < b; j++)
      for (int i = 0; i < a; i++){
	A[idx] += minrowA[i];
	idx++;
      }
    idx=0;
    for (int k = 0; k < c; k++)
      for (int j = 0; j < b; j++){
	B[idx] += mincolB[k];
	idx++;
      }
  }



  std::vector<double> pff_min_sum(int K0, int K1, int K2,  std::vector<double> & A, std::vector<double> & B, std::vector<int> & argmin_ik){

    int a=K0;
    int b=K1;
    int c=K2;

    std::vector<double> C(a*c, 0);

    int* AL = new int[a];
    int* BL = new int[c];

    int* Rowdone = new int[a];
    int* Coldone = new int[c];

    for (int i = 0; i < a; i++)
      Rowdone[i] = 0;
    for (int k = 0; k < c; k++)
      Coldone[k] = 0;

    double infinity = INF;

    // Rescale the matrices.
    double* minrowA = new double[a]; // Min for each row of A (etc.)
    double* minrowB = new double[b];
    double* mincolA = new double[b];
    double* mincolB = new double[c];

    minrowcol(A, B, a, b, b, c, minrowA, mincolB);
    normalize(A, B, a, b, c, minrowA, mincolB);
    minrowcol(B, A, b, c, a, b, minrowB, mincolA);

    static double T_static  = 0.001;
    double Cmax;

    double T;
    int e0,e1,e2;

    /// The main part of the algorithm.
    while (true) {

      // Set all active entries to infinity.
      for (int i = 0; i < a; i++)
	for (int k = 0; k < c; k++)
	  if (!Rowdone[i] && !Coldone[k])
	    C[i+k*a] = infinity;

      T = T_static;

      for (int j = 0; j < b; j++) {
	// build lists
	int numAL = 0;
	double mrB = minrowB[j];
	double mcA = mincolA[j];
	e0 = j*a;
	for (int i = 0; i < a; i++) {
	  if (!Rowdone[i] and A[e0] + mrB <= T)
	    AL[numAL++] = i;
	  e0++;
	}
	int numBL = 0;
	e1 = j;
	for (int k = 0; k < c; k++) {
	  if (!Coldone[k] and B[e1] + mcA <= T)
	    BL[numBL++] = k;
	  e1+=b; // e1 = j + k * b
	}

	for (int pA = 0; pA < numAL; pA++) {
	  int i = AL[pA];
	  for (int pB = 0; pB < numBL; pB++) {
	    int k = BL[pB];
	    e2 = i+k*a;
	    double val = A[i+j*a] + B[j+k*b];
	    if (val < C[e2]){
	      C[e2] = val;
	      argmin_ik[e2] = j;
	    }
	  }
	}
      }

      // check which rows/columns are done
      Cmax = -infinity;
      int done = 1;

      for (int i = 0; i < a; i++) {
	Rowdone[i] = 1;
	e2 = i;
	for (int k = 0; k < c; k++) {
	  if (C[e2] > T) {
	    Rowdone[i] = 0;
	    done = 0;
	  }
	  if (C[e2] > Cmax)
	    Cmax = C[e2];
	  e2+=a;
	}
      }
      e2=0;
      for (int k = 0; k < c; k++) {
	Coldone[k] = 1;
	for (int i = 0; i < a; i++) {
	  if (C[e2] > T) {
	    Coldone[k] = 0;
	    done = 0;
	  }
	  e2++;
	}
      }

      if (done)
	break;

      T_static *= 1.5;
    }

    // T grows exponentially, but we'll use its current value to help initialization next time
    if (Cmax)
      T_static = 0.75 * Cmax;
    if (T_static <= 0)
      T_static = 0.001;

    unnormalize(A, B, C, a, b, c, minrowA, mincolB);

    // Cleanup.
    delete[] AL;
    delete[] BL;
    delete[] Rowdone;
    delete[] Coldone;

    delete[] minrowA;
    delete[] minrowB;
    delete[] mincolA;
    delete[] mincolB;

    return C;
  }



  // Generate samples from normal distribution

  ///////////////////////////////////////////////////////////////
  // Author(s): Michael Zillich, Thomas Mörwald, 
  // Johann Prankl, Andreas Richtsfeld, Bence Magyar (ROS.org)
  //////////////////////////////////////////////////////////////

  double randn(double mu, double sigma) {
    static bool deviateAvailable=false;        //        flag
    static float storedDeviate;                        //        deviate from previous calculation
    double polar, rsquared, var1, var2;
    //        If no deviate has been stored, the polar Box-Muller transformation is
    //        performed, producing two independent normally-distributed random
    //        deviates.  One is stored for the next round, and one is returned.
    if (!deviateAvailable) {
      //        choose pairs of uniformly distributed deviates, discarding those
      //        that don't fall within the unit circle
      do {
	var1=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
	var2=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
	rsquared=var1*var1+var2*var2;
      } while ( rsquared>=1.0 || rsquared == 0.0);
      //        calculate polar tranformation for each deviate
      polar=sqrt(-2.0*log(rsquared)/rsquared);
      //        store first deviate and set flag
      storedDeviate=var1*polar;
      deviateAvailable=true;
      //        return second deviate
      return var2*polar*sigma + mu;
    }
    //        If a deviate is available from a previous call to this function, it is
    //        returned, and the flag is set to false.
    else {
      deviateAvailable=false;
      return storedDeviate*sigma + mu;
    }
  }

}
