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


  The following paper needed to be cited for any publication of research 
  work that uses this software.

  Huayan Wang and Daphne Koller: 
  A Fast and Exact Energy Minimization Algorithm for Cycle MRFs, 
  The 30th International Conference on Machine Learning (ICML 2013)

  Copyright 2013 Huayan Wang <huayanw@cs.stanford.edu>

*/

#ifndef CYCLE_H
#define CYCLE_H

#ifndef STDIO
#define STDIO
#include <cstdio>
#endif

namespace CycleSolver{

	typedef double   ValType;
	typedef int      IdxType;
    typedef unsigned IterType;

	///////////////////////////////////////////////////////////////////////////
	// Utility functions

	ValType minimize(const std::vector<ValType> & v, IdxType & assignment);
	void print_vector(const std::vector<IdxType> & v);
	std::vector<IdxType> rotate_cycle_assignment(const std::vector<IdxType> & a, 
			IdxType offset);
	std::vector<ValType> naive_min_sum(IdxType K0, IdxType K1, IdxType K2,  
			std::vector<ValType> & A, 
			std::vector<ValType> & B, 
			std::vector<IdxType> & argmin_ik);

	// M(X0, X1) is a (K0 x K1) matrix, return \min_X0 M(X0, X1)
	std::vector<ValType> reduce_i(IdxType K0, IdxType K1, 
			const std::vector<ValType> & M);

	// M(X0, X1) is a (K0 x K1) matrix, return \min_X1 M(X0, X1)
	std::vector<ValType> reduce_j(IdxType K0, IdxType K1, 
			const std::vector<ValType> & M);

	// tar += s*M' (tar is K0xK1, M is K1xK0)
	void add_matrixT(IdxType K0, IdxType K1, std::vector<ValType> & tar, 
			const std::vector<ValType> & M, ValType s);

	// add v(X0) to M(X0,X1) (a K0XK1 matrix)
	void add_i2ij(IdxType K0, IdxType K1, const std::vector<ValType> & v, 
			std::vector<ValType> & M, ValType factor) ;
	///////////////////////////////////////////////////////////////////////////


	class Cycle
	{
		public:
			Cycle(){};

			// generate a random cycle
			void randomCycle(IdxType, IdxType, ValType);
			// initialise cycle with specific unary and pairwise potentials. 	
			void initialiseCycle(ValType **, ValType **, int *, int);

			IdxType N; // length of cycle
			std::vector<IdxType> card;   // has length size+1

			// the solvers implemented DO support different cardinalities of variables
			// but you need to construct/read-in your own problem (as in randomCycle())

			ValType obj; // min energy
			std::vector<IdxType> assignment; // has length size+1, with the two ends agree

			void runFullMinSum(IdxType d, bool fast);
			void runFastSolver(IdxType d);
			// you get Felzenswalb's method using fast = true in FullMinSum
			// FastSolver does not give exact min-marginals
			// running both solvers writes to 'obj' and 'assignment'
			// 'd' specifies direction of message passing (or how to triangulate cycle)
			// it can be 0, 1, ..., N-1, where N is cycle size
			// it will determine which min-marginal(s) we will get as a by-product

			void freeMemory();

			std::vector< std::vector<ValType> > tp; // theta_p, pairwise energy terms

			// min-marginals
			std::vector< std::vector<ValType> > mmarg;
			// mmarg[0~N-1]: node min-marginals of variable [0~N-1]
			// mmarg[N]:   edge min-marginal (var_id[N-1], var_id[0]) // mmarg[N+1]: edge min-marginal (var_id[0], var_id[1])
			// ...
			// mmarg[2N-1]: edge min-marginal (var_id[N-2], var_id[N-1])

		private:

			std::vector<ValType> msg;
			ValType *TA_static;
			ValType *TB_static;
			IdxType L;
			IdxType Kmax;
			IdxType **backtracer; 
			IdxType* AL;
			IdxType* BL;
			ValType** mincolA;
			IdxType* OldnumAL;
			IdxType* OldnumBL;
			void allocate_memory();
			void reset_thresholds(const std::vector<IdxType> & Aidx, 
					const std::vector<IdxType> & Bidx);
			void reparameterize_singleton(IdxType i);
			void reparameterize_singleton_reverse(IdxType i);
			void fast_solver_preprocessing(IdxType last_factor_id, 
					const std::vector<IdxType> & Aidx, 
					const std::vector<IdxType> & Bidx,
					std::vector<ValType> & factor_offset, 
					ValType & global_const, 
					std::vector<ValType> & rlb,
					std::vector<std::vector<ValType> > & drlb,
					ValType** mincolA);

	};

	// implementation of Cycle member functions

	void Cycle::randomCycle(IdxType _N, IdxType K, ValType unary_strength)
	{
		N = _N;
		card.assign(N+1, K);
		tp.assign(N, std::vector<ValType>());
		for(IdxType i=0; i<N; i++)
		{
			tp[i].assign(card[i]*card[i+1], 0);

			/* pariwise potential */
			for(IterType k=0; k<tp[i].size(); k++)
				tp[i][k] = randn(0.0, 1.0);

			/* unary potential */
			if(unary_strength > 0){
				std::vector<ValType> u(card[i], 0);
				for(IterType k=0; k<u.size(); k++)
					u[k] = randn(0.0, 1.0);
				add_i2ij(card[i], card[i+1], u, tp[i], unary_strength);
			}
		}
		mmarg.assign(2*N, std::vector<ValType>()); 
		allocate_memory();
	}

	void Cycle::initialiseCycle(ValType ** unaries, ValType ** pairwise, int * n_labels, int n_nodes)
	{
		N = n_nodes;
		int max_n_labels = (int) n_labels[0];

		card.assign(N+1, 0);
		card[0]          = (int) n_labels[0];

		for(IdxType i = 1; i < N; i ++)
		{
			card[i] = (int) n_labels[i];
			if(card[i] > max_n_labels)
				max_n_labels = card[i];
		}
		card[N] = card[0];

		tp.assign(N, std::vector<ValType>());
		for(IdxType i=0; i<N; i++)
		{
			tp[i].assign(card[i]*card[i+1], 0);

			/* pariwise potential */
			for(IterType k=0; k<tp[i].size(); k++)
				tp[i][k] = pairwise[i][k];

			/* unary potential */
			std::vector<ValType> u(card[i], 0);
			for(IterType k=0; k<u.size(); k++)
				u[k] = unaries[i][k];
			add_i2ij(card[i], card[i+1], u, tp[i], 1);
		}
		mmarg.assign(2*N, std::vector<ValType>()); 
		allocate_memory();
	}

	void Cycle::runFullMinSum(IdxType d, bool fast) 
	{
		assert(d>=0 && d<N);
		std::vector< std::vector<ValType> > forward_msg(N-1, std::vector<ValType>());
		std::vector< std::vector<IdxType> > forward_argmin(N-1, std::vector<IdxType>());
		for(IdxType i=0; i<N-1; i++)
			forward_argmin[i].assign(card[d]*card[(i+d+1)%N], -1);
		mmarg[N+d] = tp[(N-1+d)%N];
		forward_msg[0] = tp[d];

		// forward pass
		for(IdxType i=1; i < N-1; i++){

			if(fast){

				// (d, d+i+1)                     (d, d+i)        (d+i, d+i+1)
				forward_msg[i] = pff_min_sum(card[d], card[(d+i)%N], card[(d+i+1)%N],
						forward_msg[i-1], 
						tp[(i+d)%N], 
						forward_argmin[i]);

			}else{

				// (d, d+i+1)                     (d, d+i)        (d+i, d+i+1)
				forward_msg[i] = naive_min_sum(card[d], card[(d+i)%N], card[(d+i+1)%N],
						forward_msg[i-1], 
						tp[(i+d)%N], 
						forward_argmin[i]);
			}
		}

		// (forward_msg[N-2])' + tp[N-1] -> (1,N)
		add_matrixT(card[(N-1+d)%N], card[d], mmarg[N+d], forward_msg[N-2], 1.0);


		// back trace for assignments
		IdxType l;
		obj = minimize(mmarg[N+d], l); // l = a[N-1] + a[0]*card[N-1]
		std::vector<IdxType> rot_assignment(N+1,-1);
		rot_assignment[N-1]=l%card[(d+N-1)%N];
		rot_assignment[0] = (l-rot_assignment[N-1])/card[(d+N-1)%N];
		rot_assignment[N] = rot_assignment[0];
		for(IdxType i=N-2; i>=1; i--)
			rot_assignment[i] = forward_argmin[i][rot_assignment[0] + rot_assignment[i+1]*card[d]];
		assignment = rotate_cycle_assignment(rot_assignment, -d);
		mmarg[(N-1+d)%N] = reduce_j(card[(N-1+d)%N], card[d], mmarg[N+d]);
		mmarg[d] = reduce_i(card[(N-1+d)%N], card[d], mmarg[N+d]);
	}


	// the Omega+ operator defined in the paper
	void Cycle::reparameterize_singleton(IdxType i) 
	{
		IdxType K0 = card[(N+i-1)%N];
		IdxType K1 = card[i];
		IdxType K2 = card[(i+1)%N];
		for(IdxType l=0; l<K1; l++){ // for each assignment of node i
			ValType min1=INF;
			ValType min2=INF;
			IdxType e0 = l;
			IdxType e1 = l*K0;
			for(IdxType k=0; k<K2; k++){
				min1 = MIN(min1, tp[i][e0]); // e0 = l+k*K1
				e0+=K1;
			}
			for(IdxType k=0; k<K0; k++){
				min2 = MIN(min2, tp[(i-1+N)%N][e1]); // e1 = k+l*K0
				e1++;
			}
			ValType u = 0.5*(min2-min1);
			e0=l;
			e1=l*K0;

			for(IdxType k=0; k<K2; k++){
				tp[i][e0] += u;
				e0+=K1;
			}
			for(IdxType k=0; k<K0; k++){
				tp[(i-1+N)%N][e1] -= u;
				e1++;
			}
		}
	}

	// the Omega- operator defined in the paper
	void Cycle::reparameterize_singleton_reverse(IdxType i) 
	{
		IdxType K0 = card[(N+i-1)%N];
		IdxType K1 = card[i];
		IdxType K2 = card[(i+1)%N];
		for(IdxType l=0; l<K1; l++){ // for each assignment of node i
			ValType min1=INF;
			ValType min2=INF;
			IdxType e0 = l;
			IdxType e1 = l*K0;
			for(IdxType k=0; k<K2; k++){
				min1 = MIN(min1, tp[i][e0]); // e0 = l+k*K1
				e0+=K1;
			}
			for(IdxType k=0; k<K0; k++){
				min2 = MIN(min2, tp[(i-1+N)%N][e1]); // e1 = k+l*K0
				e1++;
			}
			ValType u = 0.5*(min2-min1);
			e0=l;
			e1=l*K0;

			for(IdxType k=0; k<K2; k++){
				tp[i][e0] -= u;
				e0+=K1;
			}
			for(IdxType k=0; k<K0; k++){
				tp[(i-1+N)%N][e1] += u;
				e1++;
			}
		}
	}

	void Cycle::freeMemory()
	{
		for (IdxType i = 0; i < L; i++){
			delete backtracer[i];
			delete mincolA[i];
		}
		delete backtracer;
		delete AL;
		delete BL;
		delete OldnumAL;
		delete OldnumBL;
		delete TA_static;
		delete TB_static;
		delete mincolA;
	}

	void Cycle::allocate_memory()
	{
		L = N-2;
		Kmax=0;
		for(IdxType i=0; i<N; i++)
			Kmax = MAX(Kmax, card[i]);
		msg.assign(2*Kmax,0);
		for(IdxType i=0; i<L-1; i++)
			tp.push_back(std::vector<ValType>(Kmax*Kmax,INF)); 
		backtracer = new IdxType*[L];
		AL = new IdxType[Kmax];
		BL = new IdxType[Kmax];
		OldnumAL = new IdxType[Kmax];
		OldnumBL = new IdxType[Kmax];
		TA_static = new ValType[L];
		TB_static = new ValType[L];
		mincolA = new ValType*[L];
		for (IdxType i = 0; i < L; i++){
			backtracer[i] = new IdxType[Kmax*Kmax];
			TA_static[i] = -1;
			TB_static[i] = -1;
			mincolA[i] = new ValType[Kmax];
		}
	}


	// set TA_static and TB_static
	// note: this private function is not optimized for human readability
	void Cycle::reset_thresholds(const std::vector<IdxType> & Aidx, 
			const std::vector<IdxType> & Bidx)
	{
		IdxType idx;
		IdxType K0,K1;
		ValType factor_offset;
		std::vector<ValType> meanVal(L,0);
		for(IdxType s=L-1; s>0; s--){
			idx=0;
			factor_offset = INF;
			K0 = card[Bidx[s]];
			K1 = card[(Bidx[s]+1)%N];
			for(IdxType i=0; i<K1; i++){
				for(IdxType j=0; j<K0; j++){
					meanVal[s] += tp[Bidx[s]][idx];
					factor_offset = MIN(factor_offset, tp[Bidx[s]][idx]);
					idx++;
				}
			}
			meanVal[s] = meanVal[s]/(K0*K1) - factor_offset;
		}
		ValType meanValA0=0;
		ValType meanValB0=0;

		idx=0;
		K0 = card[Aidx[0]];
		K1 = card[(Aidx[0]+1)%N];
		factor_offset = INF;  
		for(IdxType i=0; i<K1; i++){
			for(IdxType j=0; j<K0; j++){
				meanValA0 += tp[Aidx[0]][idx];
				factor_offset = MIN(factor_offset, tp[Aidx[0]][idx]); // idx = j+i*K0
				idx++;
			}
		}
		meanValA0 = meanValA0/(K0*K1) - factor_offset;
		idx=0;
		K0 = card[Bidx[0]];
		K1 = card[(Bidx[0]+1)%N];
		factor_offset = INF;
		for(IdxType i=0; i<K1; i++){
			for(IdxType j=0; j<K0; j++){
				meanValB0 += tp[Bidx[0]][idx];
				factor_offset = MIN(factor_offset, tp[Bidx[0]][idx]);
				idx++;
			}
		}
		meanValB0 = meanValB0/(K0*K1) - factor_offset;
		meanVal[0] = (meanValA0+meanValB0)/2.0;

		ValType batch_size = 0.1;
		TB_static[0] = meanValB0 * batch_size;
		TA_static[0] = meanValA0 * batch_size;
		for(IdxType s=1; s<L; s++){
			TB_static[s] = meanVal[s] * batch_size;
			TA_static[s] = TA_static[s-1]+TB_static[s-1];
		}
		for(IdxType s=0; s<L; s++){
			TA_static[s] = MAX(TA_static[s], EPS);
			TB_static[s] = MAX(TB_static[s], EPS);
		}
	}

	// compute the messages in cycle edges (small delta in the paper)
	// and other stuff
	// note: this private function is not optimized for human readability
	void Cycle::fast_solver_preprocessing(IdxType last_factor_id, 
			const std::vector<IdxType> & Aidx, 
			const std::vector<IdxType> & Bidx,
			std::vector<ValType> & factor_offset, 
			ValType & global_const, 
			std::vector<ValType> & rlb,
			std::vector<std::vector<ValType> > & drlb,
			ValType** mincolA)
	{


		for(IdxType i=0; i<2*Kmax; i++)
			msg[i] = INF;
		IdxType idx=0;
		IdxType K0,K1;
		K0 = card[last_factor_id];
		K1 = card[(last_factor_id+1)%N];
		for(IdxType i=0; i<K1; i++)
			for(IdxType j=0; j<K0; j++){
				msg[j] = MIN(msg[j], tp[last_factor_id][idx]); // j+i*K0
				idx++;
			}
		for(IdxType i=0; i<K0; i++){
			rlb[L] = MIN(rlb[L], msg[i]);
			drlb[L][i] = msg[i];
		}
		for(IdxType s=L-1; s>0; s--){
			idx=0;
			factor_offset[Bidx[s]] = INF;
			K0 = card[Bidx[s]];
			K1 = card[(Bidx[s]+1)%N];
			for(IdxType i=0; i<K1; i++){
				for(IdxType j=0; j<K0; j++){
					msg[Kmax+j] = MIN(msg[Kmax+j], tp[Bidx[s]][idx] + msg[i]); 
					factor_offset[Bidx[s]] = MIN(factor_offset[Bidx[s]], tp[Bidx[s]][idx]);
					idx++;
				}
			}
			for(IdxType j=0; j<K0; j++){
				msg[j] = msg[Kmax+j] - factor_offset[Bidx[s]];
				rlb[s] = MIN(rlb[s], msg[j]);
				drlb[s][j] = msg[j];
				msg[Kmax+j] = INF;
			}
			global_const += factor_offset[Bidx[s]];
		}

		idx=0;
		K0 = card[Aidx[0]];
		K1 = card[(Aidx[0]+1)%N];
		factor_offset[Aidx[0]] = INF;  
		for(IdxType i=0; i<K1; i++){
			for(IdxType j=0; j<K0; j++){
				mincolA[0][i] = MIN(mincolA[0][i], tp[Aidx[0]][idx]);
				factor_offset[Aidx[0]] = MIN(factor_offset[Aidx[0]], tp[Aidx[0]][idx]);
				idx++;
			}
		}
		for(IdxType i=0; i<K1; i++){
			mincolA[0][i] -= factor_offset[Aidx[0]];
		}
		idx=0;
		K0 = card[Bidx[0]];
		K1 = card[(Bidx[0]+1)%N];
		factor_offset[Bidx[0]] = INF;
		for(IdxType i=0; i<K1; i++){
			for(IdxType j=0; j<K0; j++){
				msg[Kmax+j] = MIN(msg[Kmax+j], tp[Bidx[0]][idx] + msg[i]);
				factor_offset[Bidx[0]] = MIN(factor_offset[Bidx[0]], tp[Bidx[0]][idx]);
				idx++;
			}
		}
		for(IdxType j=0; j<K0; j++){
			drlb[0][j] = msg[Kmax+j] - factor_offset[Bidx[0]];
			rlb[0] = MIN(rlb[0], drlb[0][j]);
		}
		global_const += factor_offset[Aidx[0]];
		global_const += factor_offset[Bidx[0]];
	}

	// note: this function is not optimized for human readability,
	// in most cases you can use it as a black box as shown in cycle.cpp
	// however, if you insist on parsing it, it should help to
	// first try to understand the fast min-sum code in thirdparty.h,
	// which is considerably simpler, and shares many common variable names
	// and structures with this one
	void Cycle::runFastSolver(IdxType d)
	{

		ValType *TA = new ValType[L];
		ValType *TB = new ValType[L];
		IdxType last_factor_id = (N-1+d)%N;
		long final_assignment = -1;
		std::vector<ValType> factor_offset(N+L-1,0);
		std::vector<IdxType> Aidx(L,-1);
		std::vector<IdxType> Bidx(L,-1);
		Aidx[0] = d;
		for (IdxType i = 0; i < L; i++){
			Bidx[i] = (i+d+1)%N;
			for(IdxType j=0; j<Kmax; j++){
				mincolA[i][j] = INF;
			}
			if(i>0){
				Aidx[i] = N+(i-1);
				tp[Aidx[i]].assign(Kmax*Kmax,INF);
			}
		}
		std::vector<ValType> rlb(L+1,INF);
		std::vector<std::vector<ValType> > drlb(L+1, std::vector<ValType>(Kmax,INF));
		ValType global_const = 0.0;

		for(IdxType i=(d+N-1); i>=d; i--)
			reparameterize_singleton(i%N);

		if(TA_static[0] < 0)
			reset_thresholds(Aidx, Bidx);

		fast_solver_preprocessing(last_factor_id, Aidx, Bidx, 
				factor_offset, global_const, rlb, drlb, mincolA);

		ValType ub = INF;
		IdxType K0, K1, K2;
		K0 = card[(N+d-1)%N];
		K1 = card[d];

		mmarg[N+d].assign(K0*K1, INF);

		bool done;
		for(IdxType s=0; s<L; s++){
			TA[s] = TA_static[s]/1.5;
			TB[s] = TB_static[s]/1.5;
		}

		std::vector<bool> sdone(L, false);
		std::vector<bool> updated(L, false);

		// for faster index computing
		IdxType e0; // i+j*K
		IdxType e1; // j+k*K
		IdxType e2; // i+k*K
		IdxType e3; // k+i+K

		do{
			done = true;
			for(IdxType s=0; s<L; s++){
				if (sdone[s])
					continue;
				done = false;
				do{
					updated[s] = false;
					if(TB[s] < ub-rlb[s+1] || TA[s] < ub-rlb[s]){
						TB[s] = MIN(TB[s]*1.5, MAX(EPS, ub-rlb[s+1]));
						TA[s] = MIN(TA[s]*1.5, MAX(EPS, ub-rlb[s]));
					}else{
						if(s == 0){
							sdone[s] = true;
							break;
						}else if(!updated[s-1]){
							sdone[s] = sdone[s-1];
							break;
						}
					}
					K0 = card[d];
					K1 = card[Bidx[s]];
					K2 = card[(Bidx[s]+1)%N];
					for(IdxType j=0; j<K1; j++){
						IdxType numAL = 0;
						IdxType numBL = 0;
						ValType mcA = mincolA[s][j];

						// extract from entries from A
						e0 = j*K0;
						for(IdxType i=0; i<K0; i++){

							if(tp[Aidx[s]][e0] - factor_offset[Aidx[s]] <= 
									MIN(TA[s], ub-drlb[s][j])){
								AL[numAL++] = i;
							}
							e0++;
						}
						// extract from entries from B
						e1 = j;
						for(IdxType k=0; k<K2; k++){
							if(tp[Bidx[s]][e1] - factor_offset[Bidx[s]] + mcA <= 
									MIN(TB[s], ub-drlb[s+1][k])) {
								BL[numBL++] = k;
							}
							e1+=K1;
						}

						// perform partial min-sum
						for(IdxType pA=0; pA<numAL; pA++){
							IdxType i = AL[pA];
							for(IdxType pB=0; pB<numBL; pB++){
								IdxType k = BL[pB];
								e0 = i+j*K0;
								e1 = j+k*K1;
								e2 = i+k*K0;
								ValType val = tp[Aidx[s]][e0] + tp[Bidx[s]][e1] - 
									factor_offset[Aidx[s]] - factor_offset[Bidx[s]];
								if(s < L-1){ // mid of the clique chain
									if(val < tp[Aidx[s+1]][e2] && val < ub - drlb[s+1][k])
									{
										updated[s] = true;
										backtracer[s][e2] = j;
										tp[Aidx[s+1]][e2] = val;
										mincolA[s+1][k] = MIN(val, mincolA[s+1][k]);
									}
								}else{ // end of the clique chain
									e3 = k+i*K2;
									ValType lf = tp[last_factor_id][e3] - 
										factor_offset[last_factor_id];
									mmarg[N+d][e3] = MIN(mmarg[N+d][e3], val+lf);
									if(val+lf < ub){
										updated[s] = true;
										backtracer[s][e2] = j;
										final_assignment = e2;
										ub = val+lf;
									}
								}
							}
						}
					}
					if(updated[s])
						break;
					else if(TB[s] >= ub-rlb[s+1] && TA[s] >= ub-rlb[s]){
						sdone[s] = s==0?true:sdone[s-1];
						break;
					}
				}while( true ); // you should hope it will end :)
			}
		}while(!done);

		// memorize these value for solving next cycle faster
		// in a dual decomposition environment
		for(IdxType i=0; i<L; i++){
			TA_static[i] = TA[i]/1.5;
			TB_static[i] = TB[i]/1.5;
		} 

		K0 = card[(N+d-1)%N];
		K1 = card[d];
		obj = ub + global_const; // minimum energy

		//  trace back to get assignment
		std::vector<IdxType> rot_assignment(N+1,-1);
		rot_assignment[0] = final_assignment%card[d];
		rot_assignment[N-1] = (final_assignment-rot_assignment[0])/card[d];
		for(IdxType i=N-2; i>=1; i--){
			long idx = rot_assignment[0] + rot_assignment[i+1]*card[d];
			rot_assignment[i] = backtracer[i-1][idx];
		}
		rot_assignment[N] = rot_assignment[0];
		assignment = rotate_cycle_assignment(rot_assignment, -d);

		delete[] TA;
		delete[] TB;
	}

	// implementation of utility functions

	ValType minimize(const std::vector<ValType> & v, IdxType & assignment)
	{
		ValType minval = v[0];
		assignment = 0;
		for(IterType i=1; i<v.size(); i++){
			if(v[i] < minval){
				minval = v[i];
				assignment = i;
			}
		}
		return minval;
	}

	void print_vector(const std::vector<IdxType> & v)
	{
		printf("(");
		for(IterType i=0; i<v.size()-1; i++)
			printf("%d,", v[i]);
		printf("%d)\n", v[v.size()-1]);
	}

	std::vector<IdxType> rotate_cycle_assignment(const std::vector<IdxType> & a, IdxType offset)
	{
		IdxType N = a.size()-1;
		std::vector<IdxType> result(N,-1);
		for(IdxType i=0; i<N; i++)
			result[i] = a[(i+offset+N)%N];
		result.push_back(result[0]);
		return result;
	}

	std::vector<ValType> naive_min_sum(IdxType K0, IdxType K1, IdxType K2,  std::vector<ValType> & A, std::vector<ValType> & B, std::vector<IdxType> & argmin_ik){

		std::vector<ValType> C(K0*K2, INF);  
		for(IdxType i=0; i<K0; i++)
			for(IdxType j=0; j<K1; j++)
				for(IdxType k=0; k<K2; k++){
					if(C[i+k*K0] > A[i+j*K0] + B[j+k*K1]){
						C[i+k*K0] = A[i+j*K0] + B[j+k*K1];
						argmin_ik[i+k*K0] = j;
					}
				}
		return C;
	}

	// M(X0, X1) is a (K0 x K1) matrix, return \min_X0 M(X0, X1)
	std::vector<ValType> reduce_i(IdxType K0, IdxType K1, const std::vector<ValType> & M)
	{
		std::vector<ValType> result(K1,0);
		IdxType idx=0; // i+j*K0
		for(IdxType j=0; j<K1; j++){
			ValType minval = M[idx++];
			for(IdxType i=1; i<K0; i++){
				if(M[idx++] < minval){
					minval = M[idx-1];
				}
			}
			result[j] = minval;
		}
		return result;
	}

	// M(X0, X1) is a (K0 x K1) matrix, return \min_X1 M(X0, X1)
	std::vector<ValType> reduce_j(IdxType K0, IdxType K1, const std::vector<ValType> & M)
	{
		std::vector<ValType> result(K0,0);
		IdxType idx;
		for(IdxType i=0; i<K0; i++){
			ValType minval = M[i];
			idx = i + K0;  // i+j*K0
			for(IdxType j=1; j<K1; j++){
				if(M[idx] < minval){
					minval = M[idx];
				}
				idx += K0;
			}
			result[i] = minval;
		}
		return result;
	}

	// tar += s*M' (tar is K0xK1, M is K1xK0)
	void add_matrixT(IdxType K0, IdxType K1, std::vector<ValType> & tar, 
			const std::vector<ValType> & M, ValType s){
		IdxType idx=0;
		IdxType idxT;
		for(IdxType l=0; l<K1; l++){
			idxT = l;
			for(IdxType k=0; k<K0; k++){
				tar[idx] += s*M[idxT]; // k+l*K0, l+k*K1
				idx++;
				idxT += K1;
			}
		}
	}

	// add v(X0) to M(X0,X1) (a K0XK1 matrix)
	void add_i2ij(IdxType K0, IdxType K1, const std::vector<ValType> & v, 
			std::vector<ValType> & M, ValType factor) 
	{
		IdxType idx=0;
		for(IdxType j=0; j<K1; j++)
			for(IdxType i=0; i<K0; i++)
				M[idx++] += v[i]*factor; // M[i+j*K0]
	}

}

#endif
