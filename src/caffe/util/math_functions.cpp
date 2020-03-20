
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <unistd.h>
#include <sys/time.h>

//#include <immintrin.h>
//#include <pmmintrin.h>

#include <boost/thread.hpp>

//#define AAA
#define BBB
namespace caffe {

	/*
void sblas_igemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const char alpha, const char* A, const char* B, const char beta, int* C) {
			struct timeval start, end, res;
	int* C_data = C;
	const char* A_data;
	const char* B_data;
	printf("%d %d %d\n", M, N, K);
			gettimeofday(&start, NULL);
	const char* A_d = A-K;
	const char* B_d;
	char* B_t;
	//B_Transpose
	B_data = B;
	for(int i = 0; i < K; i++) {
		B_t = &B_T[0]+i;
		for(int j = 0; j < N; j++) {
			*B_t = *B_data;
			B_t+=K;
			B_data++;
		}
	}
	for(int i = 0; i < M; i++){
		int j;
		A_d += K;

		B_data = B_T;
		for(j=0; j < N-4; j+=4){
			A_data = A_d;
			int acc0 = 0;
			int acc1 = 0;
			int acc2 = 0;
			int acc3 = 0;
			int k=0;
			for(; k < K-4; k+=4){
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc0 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc0 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc0 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc2 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K))2
				acc1 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
				acc1 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
				acc1 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));

				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc2 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
				acc2 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
				acc2 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));

				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				acc3 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));
				acc3 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));
				acc3 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));

				A_data+=4;
				B_data+=4;
			}
			for(;k<K;k++) {
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				A_data++;
				B_data++;
			}
			(*(C_data)) = acc0;
			(*(C_data+1)) = acc1;
			(*(C_data+2)) = acc2;
			(*(C_data+3)) = acc3;
			C_data+=4;
			B_data+=3*K;
		}
		for(; j < N; j++){
			A_data = A_d;
			int acc = 0;
			int k=0;
			for(; k < K-4; k+=4){
				acc += (short)(*A_data) * (short)(*B_data);
				acc += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc += (short)(*(A_data+3)) * (short)(*(B_data+3));
				A_data+=4;
				B_data+=4;
			}
			for(;k<K;k++) {
				acc += (short)(*A_data) * (short)(*B_data);
				A_data++;
				B_data ++;
			}
			(*C_data) = acc;
			C_data++;
		}

		/*
		B_d = B;
		for(j = 0; j < N-4; j+=4){
			A_data = A_d;
			B_data = B_d;
			B_d+=4;
			int acc0 = 0;
			int acc1 = 0;
			int acc2 = 0;
			int acc3 = 0;
			int k=K>>2;
			for(; k!=0 ; k--){
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc1 += (short)(*A_data) * (short)(*(B_data+1));
				acc2 += (short)(*A_data) * (short)(*(B_data+2));
				acc3 += (short)(*A_data) * (short)(*(B_data+3));

				acc0 += (short)(*(A_data+1)) * (short)(*(B_data+N));
				acc1 += (short)(*(A_data+1)) * (short)(*(B_data+N+1));
				acc2 += (short)(*(A_data+1)) * (short)(*(B_data+N+2));
				acc3 += (short)(*(A_data+1)) * (short)(*(B_data+N+3));

				acc0 += (short)(*(A_data+2)) * (short)(*(B_data+2*N));
				acc1 += (short)(*(A_data+2)) * (short)(*(B_data+2*N+1));
				acc2 += (short)(*(A_data+2)) * (short)(*(B_data+2*N+2));
				acc3 += (short)(*(A_data+2)) * (short)(*(B_data+2*N+3));

				acc0 += (short)(*(A_data+3)) * (short)(*(B_data+3*N));
				acc1 += (short)(*(A_data+3)) * (short)(*(B_data+3*N+1));
				acc2 += (short)(*(A_data+3)) * (short)(*(B_data+3*N+2));
				acc3 += (short)(*(A_data+3)) * (short)(*(B_data+3*N+3));
	
				A_data+=4;
				B_data += N*4;
			}
			
#ifndef FAST_APPX
			k=K-((K>>2)<<2);
			for(;k!=0;k--) {
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc1 += (short)(*A_data) * (short)(*(B_data+1));
				acc2 += (short)(*A_data) * (short)(*(B_data+2));
				acc3 += (short)(*A_data) * (short)(*(B_data+3));
				A_data++;
				B_data += N;
			}
#endif
			(*C_data) = acc0;
			(*(C_data+1)) = acc1;
			(*(C_data+2)) = acc2;
			(*(C_data+3)) = acc3;
			C_data+=4;
		}/
		for(; j < N; j++){
			A_data = A_d;
			B_data = B_d;
			B_d++;
			int acc = 0;
			int k=0;
			for(; k < K-4; k+=4){
				acc += (short)(*A_data) * (short)(*B_data);
				acc += (short)(*(A_data+1)) * (short)(*(B_data+N));
				acc += (short)(*(A_data+2)) * (short)(*(B_data+2*N));
				acc += (short)(*(A_data+3)) * (short)(*(B_data+3*N));
				A_data+=4;
				B_data += N*4;
			}
			for(;k<K;k++) {
				acc += (short)(*A_data) * (short)(*B_data);
				A_data++;
				B_data += N;
			}
			(*C_data) = acc;
			C_data++;
		}*/
/*	}
}
*/
//#define FAST_APPX
//INT8 Edited
#ifdef AAA
/*char B_T[10000000];
void sblas_igemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const char alpha, const char* A, const char* B, const char beta, int* C) {
	int* C_data = C;
	const char* A_data;
	const char* B_data;
	const char* A_d = A;
	const char* B_d;
	char* B_t;
	//B_Transpose
	B_data = B;
	int i=0;
	for(; i < K; i++) {
		B_t = &B_T[0]+i;
		for(int j = 0; j < N; j++) {
			*B_t = *B_data;
			B_t+=K;
			B_data++;
		}
	}

	i=0;
*/
/*	for(; i < M-2; i+=2){
		int j;
		B_data = B_T;
		for(j=0; j < N-4; j+=4){
			A_data = A_d;
			int acc00 = 0;
		int acc01 = 0;
			int acc02 = 0;
			int acc03 = 0;
			int acc10 = 0;
			int acc11 = 0;
			int acc12 = 0;
			int acc13 = 0;

			int k=0;
			for(; k < K-4; k+=4){
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc00 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc00 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc00 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc01 += (short)(*A_data) * (short)(*(B_data+K));
				acc01 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
				acc01 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
				acc01 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));

				acc02 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc02 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
				acc02 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
				acc02 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));

				acc03 += (short)(*A_data) * (short)(*(B_data+3*K));
				acc03 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));
				acc03 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));
				acc03 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));

				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K+1)) * (short)(*(B_data+1));
				acc10 += (short)(*(A_data+K+2)) * (short)(*(B_data+2));
				acc10 += (short)(*(A_data+K+3)) * (short)(*(B_data+3));

				acc11 += (short)(*(A_data+K)) * (short)(*(B_data+K));
				acc11 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+K));
				acc11 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+K));
				acc11 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+K));

				acc12 += (short)(*(A_data+K)) * (short)(*(B_data+2*K));
				acc12 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+2*K));
				acc12 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+2*K));
				acc12 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+2*K));

				acc13 += (short)(*(A_data+K)) * (short)(*(B_data+3*K));
				acc13 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+3*K));
				acc13 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+3*K));
				acc13 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+3*K));

				A_data+=4;
				B_data+=4;
			}
			for(;k<K;k++) {
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc01 += (short)(*A_data) * (short)(*(B_data+K));
				acc02 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc03 += (short)(*A_data) * (short)(*(B_data+3*K));
				
				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc11 += (short)(*(A_data+K)) * (short)(*(B_data+K));
				acc12 += (short)(*(A_data+K)) * (short)(*(B_data+2*K));
				acc13 += (short)(*(A_data+K)) * (short)(*(B_data+3*K));

				A_data++;
				B_data++;
			}
			(*(C_data)) = acc00;
			(*(C_data+1)) = acc01;
			(*(C_data+2)) = acc02;
			(*(C_data+3)) = acc03;
			
			(*(C_data+N)) = acc10;
			(*(C_data+N+1)) = acc11;
			(*(C_data+N+2)) = acc12;
			(*(C_data+N+3)) = acc13;

			C_data+=4;
			B_data+=3*K;
		}
		for(; j < N; j++){
			A_data = A_d;
			int acc00 = 0;
			int acc10 = 0;
			int k=0;
			for(; k < K-4; k+=4){
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc00 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc00 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc00 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K+1)) * (short)(*(B_data+1));
				acc10 += (short)(*(A_data+K+2)) * (short)(*(B_data+2));
				acc10 += (short)(*(A_data+K+3)) * (short)(*(B_data+3));

				A_data+=4;
				B_data+=4;
			}
			for(;k<K;k++) {
				acc00 += (short)(*(A_data)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				A_data++;
				B_data ++;
			}
			(*(C_data)) = acc00;
			(*(C_data+N)) = acc10;
			C_data++;
		}
		C_data+=N;
		A_d += K*2;
	}*/ //original except
/*
	for(; i < M; i++){

		B_data = B_T;

		int j=0;
		for(; j < N-4; j+=4){
			A_data = A_d;
			int acc0 = 0;
			int acc1 = 0;
			int acc2 = 0;
			int acc3 = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc3 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));
				acc2 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));
				acc1 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));
				acc0 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc0 += (short)(*A_data) * (short)(*B_data);

				acc0 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc1 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
				acc2 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
				acc3 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));

				acc0 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc1 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
				acc2 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
				acc3 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));

				A_data+=4;
				B_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k != 0;k--) {
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				A_data++;
				B_data++;
			}
			(*(C_data)) = acc0;
			(*(C_data+1)) = acc1;
			(*(C_data+2)) = acc2;
			(*(C_data+3)) = acc3;
			C_data+=4;
			B_data+=3*K;
		}
		for(; j < N; j++){
			A_data = A_d;
			int acc = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc += (short)(*A_data) * (short)(*B_data);
				acc += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc += (short)(*(A_data+3)) * (short)(*(B_data+3));
				A_data+=4;
				B_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k!=0;k--) {
				acc += (short)(*A_data) * (short)(*B_data);
				A_data++;
				B_data ++;
			}
			(*C_data) = acc;
			C_data++;
		}
		A_d += K;
	}
}*/
#endif
#ifdef BBB
inline int _ceil(int a, int b) {
	return (a-1)/b + 1;
}
void sblas_igemm_general(const char* A, const char* B, int* C, int M, int N, int K) {
	int* C_data = C;
	const char* A_data;
	const char* B_data;
	const char* A_d = A;
	const char* B_d;
	char* B_t;
	//B_Transpose
	int i=0;

	i=0;

	for(; i < M-2; i+=2){
		int j;
		B_data = B;
		for(j=0; j < N-4; j+=4){
			A_data = A_d;
			int acc00 = 0;
		int acc01 = 0;
			int acc02 = 0;
			int acc03 = 0;
			int acc10 = 0;
			int acc11 = 0;
			int acc12 = 0;
			int acc13 = 0;

			int k=0;
			for(; k < K-4; k+=4){
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc00 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc00 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc00 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc01 += (short)(*A_data) * (short)(*(B_data+K));
				acc01 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
				acc01 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
				acc01 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));

				acc02 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc02 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
				acc02 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
				acc02 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));

				acc03 += (short)(*A_data) * (short)(*(B_data+3*K));
				acc03 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));
				acc03 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));
				acc03 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));

				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K+1)) * (short)(*(B_data+1));
				acc10 += (short)(*(A_data+K+2)) * (short)(*(B_data+2));
				acc10 += (short)(*(A_data+K+3)) * (short)(*(B_data+3));

				acc11 += (short)(*(A_data+K)) * (short)(*(B_data+K));
				acc11 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+K));
				acc11 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+K));
				acc11 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+K));

				acc12 += (short)(*(A_data+K)) * (short)(*(B_data+2*K));
				acc12 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+2*K));
				acc12 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+2*K));
				acc12 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+2*K));

				acc13 += (short)(*(A_data+K)) * (short)(*(B_data+3*K));
				acc13 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+3*K));
				acc13 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+3*K));
				acc13 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+3*K));

				A_data+=4;
				B_data+=4;
			}
			for(;k<K;k++) {
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc01 += (short)(*A_data) * (short)(*(B_data+K));
				acc02 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc03 += (short)(*A_data) * (short)(*(B_data+3*K));
				
				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc11 += (short)(*(A_data+K)) * (short)(*(B_data+K));
				acc12 += (short)(*(A_data+K)) * (short)(*(B_data+2*K));
				acc13 += (short)(*(A_data+K)) * (short)(*(B_data+3*K));

				A_data++;
				B_data++;
			}
			(*(C_data)) = acc00;
			(*(C_data+1)) = acc01;
			(*(C_data+2)) = acc02;
			(*(C_data+3)) = acc03;
			
			(*(C_data+N)) = acc10;
			(*(C_data+N+1)) = acc11;
			(*(C_data+N+2)) = acc12;
			(*(C_data+N+3)) = acc13;

			C_data+=4;
			B_data+=3*K;
		}
		for(; j < N; j++){
			A_data = A_d;
			int acc00 = 0;
			int acc10 = 0;
			int k=0;
			for(; k < K-4; k+=4){
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc00 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc00 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc00 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K+1)) * (short)(*(B_data+1));
				acc10 += (short)(*(A_data+K+2)) * (short)(*(B_data+2));
				acc10 += (short)(*(A_data+K+3)) * (short)(*(B_data+3));

				A_data+=4;
				B_data+=4;
			}
			for(;k<K;k++) {
				acc00 += (short)(*(A_data)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				A_data++;
				B_data ++;
			}
			(*(C_data)) = acc00;
			(*(C_data+N)) = acc10;
			C_data++;
		}
		C_data+=N;
		A_d += K*2;
	}

	for(; i < M; i++){

		B_data = B;

		int j=0;
		for(; j < N-4; j+=4){
			A_data = A_d;
			int acc0 = 0;
			int acc1 = 0;
			int acc2 = 0;
			int acc3 = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc3 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));
				acc2 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));
				acc1 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));
				acc0 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc0 += (short)(*A_data) * (short)(*B_data);

				acc0 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc1 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
				acc2 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
				acc3 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));

				acc0 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc1 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
				acc2 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
				acc3 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));

				A_data+=4;
				B_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k != 0;k--) {
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				A_data++;
				B_data++;
			}
			(*(C_data)) = acc0;
			(*(C_data+1)) = acc1;
			(*(C_data+2)) = acc2;
			(*(C_data+3)) = acc3;
			C_data+=4;
			B_data+=3*K;
		}
		for(; j < N; j++){
			A_data = A_d;
			int acc = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc += (short)(*A_data) * (short)(*B_data);
				acc += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc += (short)(*(A_data+3)) * (short)(*(B_data+3));
				A_data+=4;
				B_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k!=0;k--) {
				acc += (short)(*A_data) * (short)(*B_data);
				A_data++;
				B_data ++;
			}
			(*C_data) = acc;
			C_data++;
		}
		A_d += K;
	}
}
#define THREADN 8
void accum(const char* A, const char* B, int* C, int M, int N, int K, int m)
{
	const char* A_data;
	const char* A_d;
	const char* B_data;
	int* C_data;
	int i;
	int start = m*_ceil(M,THREADN), end = (m+1)*_ceil(M,THREADN);
	if(M<end) end = M;
	for(i=start;i<end; i++) {
		A_d = A + i*K;
		B_data = B;
		C_data = C + i*N;
		int j=N>>2;
		for(; j != 0; j--){
			A_data = A_d;
			int acc0 = 0;
			int acc1 = 0;
			int acc2 = 0;
			int acc3 = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc0 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc0 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc0 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc1 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
				acc1 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
				acc1 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));

				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc2 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
				acc2 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
				acc2 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));

				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				acc3 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));
				acc3 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));
				acc3 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));

				A_data+=4;
				B_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k != 0;k--) {
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				A_data++;
				B_data++;
			}
			(*(C_data)) = acc0;
			(*(C_data+1)) = acc1;
			(*(C_data+2)) = acc2;
			(*(C_data+3)) = acc3;
			C_data+=4;
			B_data+=3*K;
		}
		j=N-((N>>2)<<2);
		for(; j != 0; j--){
			A_data = A_d;
			int acc = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc += (short)(*A_data) * (short)(*B_data);
				acc += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc += (short)(*(A_data+3)) * (short)(*(B_data+3));
				A_data+=4;
				B_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k!=0;k--) {
				acc += (short)(*A_data) * (short)(*B_data);
				A_data++;
				B_data ++;
			}
			(*C_data) = acc;
			C_data++;
		}
	}
}
void accum_b(const char* A, const char* B, int* C, int M, int N, int K, int n)
{
	int Nt = (n+1)*_ceil(N,THREADN);
	int tmpj = n*_ceil(N,THREADN);
	if(N<Nt) Nt = N;

	for(int m=0;m<M;m++) {
	const char* A_data = A + m*K;
	const char* A_d = A_data;
	const char* B_data;
	int* C_data;
	int i;
	int j = tmpj;
	B_data = B + j*K;
	C_data = C + m*N + j;
	for(; j < Nt-3; j+=4){
		A_data = A_d;
		int acc0 = 0;
		int acc1 = 0;
		int acc2 = 0;
		int acc3 = 0;
		int k=K>>2;
		for(; k != 0; k--){
			acc0 += (short)(*A_data) * (short)(*B_data);
			acc0 += (short)(*(A_data+1)) * (short)(*(B_data+1));
			acc0 += (short)(*(A_data+2)) * (short)(*(B_data+2));
			acc0 += (short)(*(A_data+3)) * (short)(*(B_data+3));

			acc1 += (short)(*A_data) * (short)(*(B_data+K));
			acc1 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
			acc1 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
			acc1 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));

			acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
			acc2 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
			acc2 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
			acc2 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));

			acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
			acc3 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));
			acc3 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));
			acc3 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));

			A_data+=4;
			B_data+=4;
		}
		k=K-((K>>2)<<2);
		for(;k != 0;k--) {
			acc0 += (short)(*A_data) * (short)(*B_data);
			acc1 += (short)(*A_data) * (short)(*(B_data+K));
			acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
			acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
			A_data++;
			B_data++;
		}
		(*(C_data)) = acc0;
		(*(C_data+1)) = acc1;
		(*(C_data+2)) = acc2;
		(*(C_data+3)) = acc3;
		C_data+=4;
		B_data+=3*K;
	}
	for(; j < Nt; j++){
		A_data = A_d;
		int acc = 0;
		int k=K>>2;
		for(; k != 0; k--){
			acc += (short)(*A_data) * (short)(*B_data);
			acc += (short)(*(A_data+1)) * (short)(*(B_data+1));
			acc += (short)(*(A_data+2)) * (short)(*(B_data+2));
			acc += (short)(*(A_data+3)) * (short)(*(B_data+3));
			A_data+=4;
			B_data+=4;
		}
		k=K-((K>>2)<<2);
		for(;k!=0;k--) {
			acc += (short)(*A_data) * (short)(*B_data);
			A_data++;
			B_data++;
		}
		(*C_data) = acc;
		C_data++;
	}
	}
}

//char B_T[10000000];
boost::thread t[THREADN+1];
void sblas_igemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const char alpha, const char* A, const char* B, const char beta, int* C, char *B_T) {
	const char* B_data;
	char* B_t;
	//B_Transpose
	B_data = B;
	int i=0;
	for(; i < K; i++) {
		B_t = B_T+i;
		for(int j = 0; j < N; j++) {
			*B_t = *B_data;
			B_t+=K;
			B_data++;
		}
	}

//	i=0;

	if(M==1) {
		sblas_igemm_general(A, B_T, C, M, N, K);
	}
	else if(M>=THREADN) {
		for(i=0; i < THREADN; i++){
			t[i] = boost::thread{accum, A, B_T, C, M, N, K, i};
		}
		for(i=0; i < THREADN; i++){
			t[i].join();
		}
	}
	else if(N>=THREADN){
		int j;
		for(j=0; j < THREADN; j++){
			t[j] = boost::thread{accum_b, A, B_T, C, M, N, K, j};
		}
		for(j=0; j < THREADN; j++){
			t[j].join();
		}
	}
	else {
		sblas_igemm_general(A, B_T, C, M, N, K);
	}
}
#endif

void sblas_asymm_igemm_general(const unsigned char* A, const unsigned char* B, int* C, int M, int N, int K) {
	int* C_data = C;
	const unsigned char* A_data;
	const unsigned char* B_data;
	const unsigned char* A_d = A;
	const unsigned char* B_d;
	//B_Transpose
	int i=0;

	i=0;

	for(; i < M-2; i+=2){
		int j;
		B_data = B;
		for(j=0; j < N-4; j+=4){
			A_data = A_d;
			int acc00 = 0;
		int acc01 = 0;
			int acc02 = 0;
			int acc03 = 0;
			int acc10 = 0;
			int acc11 = 0;
			int acc12 = 0;
			int acc13 = 0;

			int k=0;
			for(; k < K-4; k+=4){
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc00 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc00 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc00 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc01 += (short)(*A_data) * (short)(*(B_data+K));
				acc01 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
				acc01 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
				acc01 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));

				acc02 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc02 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
				acc02 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
				acc02 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));

				acc03 += (short)(*A_data) * (short)(*(B_data+3*K));
				acc03 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));
				acc03 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));
				acc03 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));

				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K+1)) * (short)(*(B_data+1));
				acc10 += (short)(*(A_data+K+2)) * (short)(*(B_data+2));
				acc10 += (short)(*(A_data+K+3)) * (short)(*(B_data+3));

				acc11 += (short)(*(A_data+K)) * (short)(*(B_data+K));
				acc11 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+K));
				acc11 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+K));
				acc11 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+K));

				acc12 += (short)(*(A_data+K)) * (short)(*(B_data+2*K));
				acc12 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+2*K));
				acc12 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+2*K));
				acc12 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+2*K));

				acc13 += (short)(*(A_data+K)) * (short)(*(B_data+3*K));
				acc13 += (short)(*(A_data+K+1)) * (short)(*(B_data+1+3*K));
				acc13 += (short)(*(A_data+K+2)) * (short)(*(B_data+2+3*K));
				acc13 += (short)(*(A_data+K+3)) * (short)(*(B_data+3+3*K));

				A_data+=4;
				B_data+=4;
			}
			for(;k<K;k++) {
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc01 += (short)(*A_data) * (short)(*(B_data+K));
				acc02 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc03 += (short)(*A_data) * (short)(*(B_data+3*K));
				
				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc11 += (short)(*(A_data+K)) * (short)(*(B_data+K));
				acc12 += (short)(*(A_data+K)) * (short)(*(B_data+2*K));
				acc13 += (short)(*(A_data+K)) * (short)(*(B_data+3*K));

				A_data++;
				B_data++;
			}
			(*(C_data)) = acc00;
			(*(C_data+1)) = acc01;
			(*(C_data+2)) = acc02;
			(*(C_data+3)) = acc03;
			
			(*(C_data+N)) = acc10;
			(*(C_data+N+1)) = acc11;
			(*(C_data+N+2)) = acc12;
			(*(C_data+N+3)) = acc13;

			C_data+=4;
			B_data+=3*K;
		}
		for(; j < N; j++){
			A_data = A_d;
			int acc00 = 0;
			int acc10 = 0;
			int k=0;
			for(; k < K-4; k+=4){
				acc00 += (short)(*A_data) * (short)(*B_data);
				acc00 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc00 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc00 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K+1)) * (short)(*(B_data+1));
				acc10 += (short)(*(A_data+K+2)) * (short)(*(B_data+2));
				acc10 += (short)(*(A_data+K+3)) * (short)(*(B_data+3));

				A_data+=4;
				B_data+=4;
			}
			for(;k<K;k++) {
				acc00 += (short)(*(A_data)) * (short)(*B_data);
				acc10 += (short)(*(A_data+K)) * (short)(*B_data);
				A_data++;
				B_data ++;
			}
			(*(C_data)) = acc00;
			(*(C_data+N)) = acc10;
			C_data++;
		}
		C_data+=N;
		A_d += K*2;
	}

	for(; i < M; i++){

		B_data = B;

		int j=0;
		for(; j < N-4; j+=4){
			A_data = A_d;
			int acc0 = 0;
			int acc1 = 0;
			int acc2 = 0;
			int acc3 = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc3 += (short)(*(A_data+3)) * (short)(*(B_data+3+3*K));
				acc2 += (short)(*(A_data+3)) * (short)(*(B_data+3+2*K));
				acc1 += (short)(*(A_data+3)) * (short)(*(B_data+3+K));
				acc0 += (short)(*(A_data+3)) * (short)(*(B_data+3));

				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc0 += (short)(*A_data) * (short)(*B_data);

				acc0 += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc1 += (short)(*(A_data+1)) * (short)(*(B_data+1+K));
				acc2 += (short)(*(A_data+1)) * (short)(*(B_data+1+2*K));
				acc3 += (short)(*(A_data+1)) * (short)(*(B_data+1+3*K));

				acc0 += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc1 += (short)(*(A_data+2)) * (short)(*(B_data+2+K));
				acc2 += (short)(*(A_data+2)) * (short)(*(B_data+2+2*K));
				acc3 += (short)(*(A_data+2)) * (short)(*(B_data+2+3*K));

				A_data+=4;
				B_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k != 0;k--) {
				acc0 += (short)(*A_data) * (short)(*B_data);
				acc1 += (short)(*A_data) * (short)(*(B_data+K));
				acc2 += (short)(*A_data) * (short)(*(B_data+2*K));
				acc3 += (short)(*A_data) * (short)(*(B_data+3*K));
				A_data++;
				B_data++;
			}
			(*(C_data)) = acc0;
			(*(C_data+1)) = acc1;
			(*(C_data+2)) = acc2;
			(*(C_data+3)) = acc3;
			C_data+=4;
			B_data+=3*K;
		}
		for(; j < N; j++){
			A_data = A_d;
			int acc = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc += (short)(*A_data) * (short)(*B_data);
				acc += (short)(*(A_data+1)) * (short)(*(B_data+1));
				acc += (short)(*(A_data+2)) * (short)(*(B_data+2));
				acc += (short)(*(A_data+3)) * (short)(*(B_data+3));
				A_data+=4;
				B_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k!=0;k--) {
				acc += (short)(*A_data) * (short)(*B_data);
				A_data++;
				B_data ++;
			}
			(*C_data) = acc;
			C_data++;
		}
		A_d += K;
	}
}

void sblas_asymm_igemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const char alpha, const char* A, const char* B, const char beta, int* C, char *B_T) {
	const char* B_data;
	char* B_t;
	//B_Transpose
	B_data = B;
	int i=0;
	for(; i < K; i++) {
		B_t = B_T+i;
		for(int j = 0; j < N; j++) {
			*B_t = *B_data;
			B_t+=K;
			B_data++;
		}
	}

	sblas_asymm_igemm_general((unsigned char*)A, (unsigned char*)B_T, C, M, N, K);
}
/*
__attribute__ ((aligned(16))) int A_T[10000000];
__attribute__ ((aligned(16))) int B_T[10000000];
void sblas_igemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const char alpha, const char* A, const char* B, const char beta, int* C) {
			struct timeval start, end, res;
	int* C_data = C;
	const char* A_data;
	const char* B_data;
	printf("%d %d %d\n", M, N, K);
			gettimeofday(&start, NULL);
	const char* A_d = A;
	const char* B_d;
	int* A_t_data;
	int* B_t_data;
	int* A_t;
	int* B_t;
	int* A_t_d = A_T;
	int* B_t_d;
	//B_Transpose
	B_data = B;
	int i=0;
	for(; i < K; i++) {
		B_t = &B_T[0]+i;
		for(int j = 0; j < N; j++) {
			*B_t = *B_data;
			B_t+=K;
			B_data++;
		}
	}
	A_data = A;
	A_t = &A_T[0];
	i=0;
	for(; i < M; i++) {
		for(int j = 0; j < K; j++) {
			*A_t = *A_data;
			A_t++;
			A_data++;
		}
	}

	i=0;
	for(; i < M; i++){

		B_t_data = B_T;

		int j=N>>2;
		for(; j != 0; j--){
			A_t_data = A_t_d;
			int acc0 = 0;
			int acc1 = 0;
			int acc2 = 0;
			int acc3 = 0;
			int k=K>>2;
			for(; k != 0; k--){
				__m128i va, vb, vc;
				va = _mm_store_si128((__m128i *)A_t_data);
				vb = _mm_store_si128((__m128i *)B_t_data);
//				vc = _mm_multi_*/
/*				acc0 += (short)(*A_t_data) * (short)(*B_t_data);
				acc0 += (short)(*(A_t_data+1)) * (short)(*(B_t_data+1));
				acc0 += (short)(*(A_t_data+2)) * (short)(*(B_t_data+2));
				acc0 += (short)(*(A_t_data+3)) * (short)(*(B_t_data+3));

				acc1 += (short)(*A_t_data) * (short)(*(B_t_data+K));
				acc1 += (short)(*(A_t_data+1)) * (short)(*(B_t_data+1+K));
				acc1 += (short)(*(A_t_data+2)) * (short)(*(B_t_data+2+K));
				acc1 += (short)(*(A_t_data+3)) * (short)(*(B_t_data+3+K));

				acc2 += (short)(*A_t_data) * (short)(*(B_t_data+2*K));
				acc2 += (short)(*(A_t_data+1)) * (short)(*(B_t_data+1+2*K));
				acc2 += (short)(*(A_t_data+2)) * (short)(*(B_t_data+2+2*K));
				acc2 += (short)(*(A_t_data+3)) * (short)(*(B_t_data+3+2*K));

				acc3 += (short)(*A_t_data) * (short)(*(B_t_data+3*K));
				acc3 += (short)(*(A_t_data+1)) * (short)(*(B_t_data+1+3*K));
				acc3 += (short)(*(A_t_data+2)) * (short)(*(B_t_data+2+3*K));
				acc3 += (short)(*(A_t_data+3)) * (short)(*(B_t_data+3+3*K));
*/
/*				A_t_data+=4;
				B_t_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k != 0;k--) {
				acc0 += (short)(*A_t_data) * (short)(*B_t_data);
				acc1 += (short)(*A_t_data) * (short)(*(B_t_data+K));
				acc2 += (short)(*A_t_data) * (short)(*(B_t_data+2*K));
				acc3 += (short)(*A_t_data) * (short)(*(B_t_data+3*K));
				A_t_data++;
				B_t_data++;
			}
			(*(C_data)) = acc0;
			(*(C_data+1)) = acc1;
			(*(C_data+2)) = acc2;
			(*(C_data+3)) = acc3;
			C_data+=4;
			B_t_data+=3*K;
		}
		j=N-((N>>2)<<2);
		for(; j != 0; j--){
			A_t_data = A_t_d;
			int acc = 0;
			int k=K>>2;
			for(; k != 0; k--){
				acc += (short)(*A_t_data) * (short)(*B_t_data);
				acc += (short)(*(A_t_data+1)) * (short)(*(B_t_data+1));
				acc += (short)(*(A_t_data+2)) * (short)(*(B_t_data+2));
				acc += (short)(*(A_t_data+3)) * (short)(*(B_t_data+3));
				A_t_data+=4;
				B_t_data+=4;
			}
			k=K-((K>>2)<<2);
			for(;k!=0;k--) {
				acc += (short)(*A_t_data) * (short)(*B_t_data);
				A_t_data++;
				B_t_data ++;
			}
			(*C_data) = acc;
			C_data++;
		}
		A_t_d += K;
	}

			gettimeofday(&end, NULL);
			timersub(&end, &start, &res);
			printf("ingemm time: %lf\n", (float)res.tv_sec*1000.0 + (float)res.tv_usec/1000.0);
}
*/
//INT8 Edited
void sblas_intgemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const int alpha, const int* A, const float* B, const int beta, int* C) {
	int* C_data = C;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			const int* A_data = A + K*i;
			const float* B_data = B + j;
			int acc = 0;
			for(int k = 0; k < K; k++){
				acc += (*A_data) * (*B_data);
				A_data++;
				B_data += N;
			}
			(*C_data) = acc + (*C_data) * beta;
			C_data++;
		}
	}
}

void caffe_cpu_asymm_offset(const char* in_c, const char* w_c, int* out, const unsigned char in_zero_point, const unsigned char* w_zero_point, int M, int N, int K) {
	const unsigned char* in = (const unsigned char*) in_c;
	const unsigned char* w = (const unsigned char*) w_c;
	int* in_sum = (int*)malloc(sizeof(int)*N);
	int* w_offset = (int*)malloc(sizeof(int)*M);

	//in_offset
	for(int i=0;i<N;i++) {
		in_sum[i] = 0;
		for(int j=0;j<K;j++) {
			in_sum[i] += in[i + j*N];
		}
	}

	//if(in_zero_point) {
		//w_offset
		for(int i=0;i<M;i++) {
			w_offset[i] = 0;
			for(int j=0;j<K;j++) {
				w_offset[i] += w[i*K + j];
			}
			w_offset[i] *= in_zero_point;
		}
	//}

	for(int i=0;i<M;i++) {
		for(int j=0;j<N;j++) {
			out[i*N + j] -= in_sum[j] * w_zero_point[i];
			//if(in_zero_point) {
				out[i*N + j] -= w_offset[i];
				out[i*N + j] += K * in_zero_point * w_zero_point[i];
			//}
	//		printf("%d %d %d\n",in_sum[j] * w_zero_point[i],w_offset[i], in_zero_point * w_zero_point[i]);
		}
	}

	free(w_offset);
	free(in_sum);
/*
	printf("weight\n");
	for(int i=0;i<M;i++) {
		for(int j=0;j<K;j++) {
			printf("%d ", w[i*K + j]);
		}
	}
	printf("\n");
	printf("zero: %d\n", in_zero_point);
	for(int i=0;i<M;i++) {
		printf("%d ", w_zero_point[i]);
	}
	printf("\n");*/
}

//INT8 Edited
void caffe_cpu_cgemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const char alpha, const char* A, const char* B, const char beta,
    int* C, char* B_T) {
  // char gemm
  sblas_igemm(TransA, TransB, M, N, K, alpha, A, B, beta, C, B_T);
}
//INT8 Edited
void caffe_cpu_asymm_cgemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const char alpha, const char* A, const char* B, const char beta,
    int* C, char* B_T) {
  // char gemm
  sblas_asymm_igemm(TransA, TransB, M, N, K, alpha, A, B, beta, C, B_T);
}
//INT8 Edited
template<>
void caffe_cpu_igemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const int alpha, const int* A, const float* B, const int beta,
    int* C) {
  // char gemm
  sblas_intgemm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}
template<>
void caffe_cpu_igemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const int alpha, const int* A, const double* B, const int beta,
    int* C) {
  // char gemm
}

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

//INT8 Edited
template <>
void caffe_axpy<char>(const int N, const char alpha, const char* X,
    char* Y) {
	LOG(FATAL)<<"no implemted INT8 caffe_axpy";
}


template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}


//INT8 Edited
template void caffe_set<char>(const int N, const char alpha, char* Y);
template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

//INT8 Edited
template void caffe_copy<char>(const int N, const char* X, char* Y);
template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

//INT8 Edited
template <>
void caffe_scal<char>(const int N, const char alpha, char *X) {
	LOG(FATAL)<<"no implemted INT8 caffe_scal";
	X = 0;
}

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
  vsSqrt(n, a, y);
}

template <>
void caffe_sqrt<double>(const int n, const double* a, double* y) {
  vdSqrt(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

//INT8 Edited
//Spetialization for INT8
template<> void caffe_rng_uniform<char>(const int n, const char a, const char b,
                              char* r) {
	CHECK_GE(n, 0);
	CHECK(r);
	CHECK_LE(a, b);
	float a_ = (float)((int)a);
	float b_ = (float)((int)b);
	boost::uniform_real<float> random_distribution(a_, caffe_nextafter<float>(b_));
	boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
		variate_generator(caffe_rng(), random_distribution);
	for (int i = 0; i < n; ++i) {
		int r_ = (int)(variate_generator());
		r[i] = (r_>127)?(127):((r_<-128)?(-128):(r_));
	}
}
template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

//INT8 Edited
//Spetialization for INT8
template<> void caffe_rng_gaussian<char>(const int n, const char a,
                               const char sigma, char* r) {
  	CHECK(r);
	char _sigma = 1;
  	CHECK_GT(_sigma, 0);
	float a_ = (float)((int)a);
	float sigma_ = (float)((int)_sigma);
  	boost::normal_distribution<float> random_distribution(a_, sigma_);
  	boost::variate_generator<caffe::rng_t*, boost::normal_distribution<float> >
  	    variate_generator(caffe_rng(), random_distribution);
  	for (int i = 0; i < n; ++i) {
		int r_ = (int)(variate_generator());
		r[i] = (r_>127)?(127):((r_<-128)?(-128):(r_));
  	}
}
template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

//INT8 Edited
//Spetialization for INT8
template<> void caffe_rng_bernoulli<char>(const int n, const char p, int* r){
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  float p_ = (float)((int)p);
  boost::bernoulli_distribution<float> random_distribution(p_);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<float> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}


template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

//INT8 Edited
template <>
char caffe_cpu_strided_dot<char>(const int n, const char* x, const int incx,
    const char* y, const int incy) {
  LOG(FATAL)<<"no implemted INT8 caffe_strided_dot";
  return 0;
}

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

//INT8 Edited
template
char caffe_cpu_dot<char>(const int n, const char* x, const char* y);

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

//INT8 Edited
template <>
char caffe_cpu_asum<char>(const int n, const char* x) {
  LOG(FATAL)<<"no implemted INT8 caffe_asum";
  return 0;
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

}  // namespace caffe
