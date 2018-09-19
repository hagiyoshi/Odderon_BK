#include <cufft.h> 
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <cctype>
#include <complex>
#include <functional>

#include "Parameters.h"


__global__ void Integration_BK_direct(cuDoubleComplex* integrated, cuDoubleComplex* S_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.

		cuDoubleComplex complex_zero = make_cuDoubleComplex(0.0, 0.0);
		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
				//if r-z is out of the region then we take the S(r-z) =0.
				if((j - m + N / 2)<0|| (j - m + N / 2)>N-1|| (i - n + N / 2)<0|| (i - n + N / 2)>N-1){
					//trV= - S(r)
					trV_V = cuCsub(trV_V,
						S_matrix[j * N + i]);
				}
				else {
					//trV=S(r-z)
					trV_V = cuCadd(trV_V,
						S_matrix[(j - m + N / 2) * N + i - n + N / 2]);
					//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
					trV_V = cuCmul(trV_V,
						cuConj(S_matrix[m * N + n]));
					//trV=S(r-z)*S(-z) - S(r)
					trV_V = cuCsub(trV_V,
						S_matrix[j * N + i]);
				}

				cuDoubleComplex coeff = make_cuDoubleComplex(
					simpson1*simpson2
					*(x_1[j*N+i]* x_1[j*N + i]+ y_1[j*N + i] * y_1[j*N + i])
					/((x_1[m*N + n] - x_1[j*N + i])*(x_1[m*N + n] - x_1[j*N + i])+ (y_1[m*N + n] - y_1[j*N + i])*(y_1[m*N + n] - y_1[j*N + i]))
					/ (x_1[m*N + n] * x_1[m*N + n]+ y_1[m*N + n] * y_1[m*N+n]),
					0.0
				);

				if(((x_1[m*N + n] - x_1[j*N + i])*(x_1[m*N + n] - x_1[j*N + i]) 
						+ (y_1[m*N + n] - y_1[j*N + i])*(y_1[m*N + n] - y_1[j*N + i])) < 1.0e-12
					|| (x_1[m*N + n] * x_1[m*N + n] + y_1[m*N + n] * y_1[m*N+n]) < 1.0e-12){
					coeff = make_cuDoubleComplex(0.0,0.0);
				}

				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h*ALPHA_S_BAR/2.0/Pi, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


void Integration_in_BK_equation(std::complex<double>* Smatrix_in, std::complex<double>* Integrated_out)
{


	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h * NX / 1.0, xmin = -h * NX / 2.0, ymin = -h * NX / 2.0;
	double   *x = new double[N*N], *y = new double[N*N];
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i * h;
			y[NX*j + i] = ymin + j * h;
		}
	}

	// Allocate arrays on the device
	double  *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *S_matrix_d;
	cudaMalloc((void**)&S_matrix_d, sizeof(cuDoubleComplex) * N*N);
	cudaMemcpy(S_matrix_d, Smatrix_in, sizeof(std::complex<double>) * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);
	
	Integration_BK_direct <<<dimGrid, dimBlock >>> (Integrated_d, S_matrix_d, x_d, y_d, h, 0, N, N);

	cudaMemcpy(Integrated_out, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	cudaFree(x_d);
	cudaFree(y_d);;
	cudaFree(Integrated_d);
	cudaFree(S_matrix_d);
	delete[](x);
	delete[](y);
}