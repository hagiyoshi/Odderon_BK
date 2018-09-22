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

//	Returns of acos
//		Result will be in radians, in the interval[0, ƒÎ] for x inside[-1, +1].
//		acos(1) returns + 0.
//		acos(x) returns NaN for x outside[-1, +1].

//input S_matrix NX*NPHI, x_1 NX, p_1 PHI;
__device__ cuDoubleComplex linear_interpolation_Smatrix(cuDoubleComplex* S_matrix,
	double* x_1, double* p_1,double x,double p)
{
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX *3.0/ 4.0, ymin = 0.0;
	int i_x=0;
	int i_p = 0;

	for (int i=0;i<NX-1;i++)
	{
		//double dx = x_1[i];
		double dx = double(i)*h + xmin;
		double diffdx = dx - x;
		//printf("i %d \n", i);
		if (0.0 > diffdx){ i_x = i;
		//printf(" if in dx - x %.3e ,i_x %d \n", diffdx , i_x);
		}
		else { 
			//printf("finish dx - x %.3e ,i_x %d \n", diffdx, i_x); 
		continue; }
	}

	for (int i = 0; i < NPHI-1; i++)
	{
		//double dp = p_1[NX*i];
		double dp = double(i)*h_theta;
		double diffdp = dp - p;
		if (0.0 > diffdp) { i_p = i; }
		else { continue; }
	}

	if (x > x_1[NX-1] || p > 2.0*Pi || x < x_1[0] || p < ymin)
	{
		printf( "out of range \n");
		assert(1);
	}

	//printf("i_x %d , i_p %d ,x_1[i_x] %.3e , p_1[i_p] %.3e , x %.3e , p %.3e \n" , i_x, i_p, x_1[i_x], p_1[NX*i_p], x, p);
	//bilinear interpolation
	double t = (x - x_1[i_x]) / (x_1[i_x + 1] - x_1[i_x]);
	double u = (p - p_1[NX*i_p]) / (p_1[NX*(i_p + 1)] - p_1[NX*i_p]);

	cuDoubleComplex interpolate;

	interpolate = make_cuDoubleComplex(
		(1.0 - t)*(1.0 - u)*S_matrix[i_p*NX + i_x].x + t * (1.0 - u)*S_matrix[i_p*NX + i_x + 1].x
		+ t * u*S_matrix[(i_p + 1)*NX + i_x + 1].x + (1.0 - t)*u*S_matrix[(i_p + 1)*NX + i_x].x   ,
		(1.0 - t)*(1.0 - u)*S_matrix[i_p*NX + i_x].y + t * (1.0 - u)*S_matrix[i_p*NX + i_x + 1].y
		+ t * u*S_matrix[(i_p + 1)*NX + i_x + 1].y + (1.0 - t)*u*S_matrix[(i_p + 1)*NX + i_x].y
	);
	if (interpolate.x > 1.0) {
		interpolate = make_cuDoubleComplex(
			1.0,
			(1.0 - t)*(1.0 - u)*S_matrix[i_p*NX + i_x].y + t * (1.0 - u)*S_matrix[i_p*NX + i_x + 1].y
			+ t * u*S_matrix[(i_p + 1)*NX + i_x + 1].y + (1.0 - t)*u*S_matrix[(i_p + 1)*NX + i_x].y
		);
	}
	else if (interpolate.y < 0.0) {
		interpolate = make_cuDoubleComplex(
			0.0,
			(1.0 - t)*(1.0 - u)*S_matrix[i_p*NX + i_x].y + t * (1.0 - u)*S_matrix[i_p*NX + i_x + 1].y
			+ t * u*S_matrix[(i_p + 1)*NX + i_x + 1].y + (1.0 - t)*u*S_matrix[(i_p + 1)*NX + i_x].y
		);
	}

	return interpolate;
}

__device__ cuDoubleComplex linear_interpolation_Smatrix_fowardonly(cuDoubleComplex* S_matrix,
	double* x_1, double* p_1, double x, double p)
{
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX *3.0 / 4.0, ymin = 0.0;
	int i_x = 0;
	int i_p = 0;

	for (int i = 0; i < NX - 1; i++)
	{
		//double dx = x_1[i];
		double dx = double(i)*h + xmin;
		double diffdx = dx - x;
		//printf("i %d \n", i);
		if (0.0 > diffdx) {
			i_x = i;
			//printf(" if in dx - x %.3e ,i_x %d \n", diffdx , i_x);
		}
		else {
			//printf("finish dx - x %.3e ,i_x %d \n", diffdx, i_x); 
			continue;
		}
	}

	for (int i = 0; i < NPHI - 1; i++)
	{
		//double dp = p_1[NX*i];
		double dp = double(i)*h_theta;
		double diffdp = dp - p;
		if (0.0 > diffdp) { i_p = i; }
		else { continue; }
	}

	if (x > x_1[NX - 1] || p > 2.0*Pi || x < x_1[0] || p < ymin)
	{
		printf("out of range \n");
		assert(1);
	}

	//printf("i_x %d , i_p %d ,x_1[i_x] %.3e , p_1[i_p] %.3e , x %.3e , p %.3e \n" , i_x, i_p, x_1[i_x], p_1[NX*i_p], x, p);
	//bilinear interpolation
	double t = (x - x_1[i_x]) / (x_1[i_x + 1] - x_1[i_x]);
	double u = (p - p_1[NX*i_p]) / (p_1[NX*(i_p + 1)] - p_1[NX*i_p]);

	cuDoubleComplex interpolate;

	interpolate = make_cuDoubleComplex(
		S_matrix[i_p*NX + i_x].x,
		S_matrix[i_p*NX + i_x].y
	);

	return interpolate;
}


__device__ double test_for_if(double* x_1, double* p_1, double x, double p)
{
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX *3.0 / 4.0, ymin = 0.0;
	int i_x = 0;
	int i_p = 0;

	for (int i = 0; i < NX - 1; i++)
	{
		//double dx = x_1[i];
		double dx = double(i)*h + xmin;
		double diffdx = dx - x;
		printf("i %d \n", i);
		if (0.0 > diffdx) {
			i_x = i;
			printf(" if in dx - x %.3e ,i_x %d \n", diffdx, i_x);
		}
		else { printf("finish dx - x %.3e ,i_x %d \n", diffdx, i_x); continue; }
	}

	for (int i = 0; i < NPHI - 1; i++)
	{
		//double dp = p_1[NX*i];
		double dp = double(i)*h_theta;
		double diffdp = dp - p;
		if (0.0 > diffdp) { i_p = i; }
		else { continue; }
	}

	return x - x_1[i_x];
}

__global__ void test_dayo(double* x_1, double* p_1)
{

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * NX + i;
	if (i < NX && j < NPHI) {
		double h = 1.0*LATTICE_SIZE / NX;
		double h_theta = 2.0*Pi / NPHI;
		double result = 0;
		//for (int i = 0; i < NX; i++) {
		int i = -2;
			result = test_for_if(x_1, p_1, i*h, i*h_theta);
		//}
	
	
	}

}

//input S_matrix NX*NPHI, x_1 NX, p_1 PHI;
__device__ cuDoubleComplex linear_interpolation_Smatrix_test(cuDoubleComplex* S_matrix,
	double* x_1, double* p_1, double x, double p)
{
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX* 3.0/ 4.0, ymin = 0.0;
	int i_x = 0;
	int i_p = 0;

	for (int i = 0; i < NX - 1; i++)
	{
		if (x < x_1[i]) { i_x = i; }
	}

	for (int i = 0; i < NPHI - 1; i++)
	{
		if (p < p_1[NX*i]) { i_p = i; }
	}

	if (x > xmax || p > 2.0*Pi || x < xmin || p < ymin)
	{
		printf("out of range \n");
		assert(1);
	}

	//bilinear interpolation
	double t = (x - x_1[i_x]) / (x_1[i_x + 1] - x_1[i_x]);
	double u = (p - p_1[NX*i_p]) / (p_1[NX*(i_p + 1)] - p_1[NX*i_p]);

	cuDoubleComplex interpolate;

	interpolate = make_cuDoubleComplex(
		(1.0 - t)*(1.0 - u)*S_matrix[i_p*NX + i_x].x + t * (1.0 - u)*S_matrix[i_p*NX + i_x + 1].x
		+ t * u*S_matrix[(i_p + 1)*NX + i_x + 1].x + (1.0 - t)*u*S_matrix[(i_p + 1)*NX + i_x].x,
		(1.0 - t)*(1.0 - u)*S_matrix[i_p*NX + i_x].y + t * (1.0 - u)*S_matrix[i_p*NX + i_x + 1].y
		+ t * u*S_matrix[(i_p + 1)*NX + i_x + 1].y + (1.0 - t)*u*S_matrix[(i_p + 1)*NX + i_x].y
	);

	return interpolate;
}


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


__global__ void Integration_BK_logscale_direct(cuDoubleComplex* integrated, cuDoubleComplex* S_matrix,
	double* x_1, double* y_1, double h, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < NPHI) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.

		double   xmax = h * NX / 4.0, xmin = -h * NX* 3.0 / 4.0, ymin = 0.0;
		double h_theta = 2.0*Pi / NPHI;
		cuDoubleComplex complex_zero = make_cuDoubleComplex(0.0, 0.0);
		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = 0; m < NPHI; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - 0;


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
				double r_z = 1.0 / 2.0*log(exp(2.0*x_1[j * N + i]) + exp(2.0*x_1[m * N + n])
											- 2.0*exp(x_1[j * N + i] + x_1[m * N + n])*cos(y_1[j * N + i] - y_1[m * N + n]));
				double r_z2 = exp(2.0*x_1[j * N + i]) + exp(2.0*x_1[m * N + n])
					- 2.0*exp(x_1[j * N + i] + x_1[m * N + n])*cos(y_1[j * N + i] - y_1[m * N + n]);
				double angletocos = (exp(x_1[j * N + i])*cos(y_1[j * N + i]) - exp(x_1[m * N + n]) * cos(y_1[m * N + n])) / sqrt(r_z2);
				if (angletocos >= 1.0) {
					//printf("cos is larger than 1 cos if %.6e\n" , angletocos-1.0);
					angletocos = 1.0;
				}
				else if (angletocos <= -1.0) {
					//printf("cos is smaller than -1 cos if %.6e\n", angletocos+1.0);
					angletocos = -1.0;
				}
				//if r-z is out of the region then we take the S(r-z) =0.
				if (r_z < xmin ) {
					cuDoubleComplex unit = make_cuDoubleComplex(1.0,0.0);
					//trV=S(r-z)
					trV_V = cuCadd(trV_V,
						unit
					);
					//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
					trV_V = cuCmul(trV_V,
						cuConj(S_matrix[m * N + n]));
					//trV= - S(r)
					trV_V = cuCsub(trV_V,
						S_matrix[j * N + i]);
				}
				else if (r_z > xmax - h) {

					cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
					//trV=S(r-z)
					trV_V = cuCadd(trV_V,
						zero
					);
					//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
					trV_V = cuCmul(trV_V,
						cuConj(S_matrix[m * N + n]));
					//trV= - S(r)
					trV_V = cuCsub(trV_V,
						S_matrix[j * N + i]);
				}
				else {
				//	Returns of acos
				//		Result will be in radians, in the interval[0, ƒÎ] for x inside[-1, +1].
				//		acos(1) returns + 0.
				//		acos(x) returns NaN for x outside[-1, +1].

					//trV=S(r-z)
					trV_V = cuCadd(trV_V,
						linear_interpolation_Smatrix(S_matrix,x_1,y_1,r_z,acos(angletocos) )
					);
					//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
					trV_V = cuCmul(trV_V,
						cuConj(S_matrix[m * N + n]));
					//trV=S(r-z)*S(-z) - S(r)
					trV_V = cuCsub(trV_V,
						S_matrix[j * N + i]);
				}
				//Caution!!! nan * 0 = nan
				if (( (x_1[j * N + i] - x_1[m * N + n])*(x_1[j * N + i] - x_1[m * N + n]) < 1.0e-10 &&
					(y_1[j * N + i] - y_1[m * N + n])*(y_1[j * N + i] - y_1[m * N + n]) < 1.0e-10 ) || r_z2<0.0) {
					trV_V = make_cuDoubleComplex(
						0.0
						,
						0.0
					);
				}

				cuDoubleComplex coeff = make_cuDoubleComplex(
					simpson1*simpson2
					*exp(2.0*x_1[j*N + i])
					/ r_z2,
					0.0
				);

				if ((x_1[j * N + i] - x_1[m * N + n])*(x_1[j * N + i] - x_1[m * N + n]) < 1.0e-10 &&
					(y_1[j * N + i] - y_1[m * N + n])*(y_1[j * N + i] - y_1[m * N + n]) < 1.0e-10) {
					coeff = make_cuDoubleComplex(
						0.0
						,
						0.0
					);
				}

				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h_theta*ALPHA_S_BAR / 2.0 / Pi, 0.0);
		
		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void Integration_BK_logscale_direct_test(cuDoubleComplex* integrated, cuDoubleComplex* S_matrix,
	double* x_1, double* y_1, double h, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < NPHI) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.

		double   xmax = h * NX / 4.0, xmin = -h * NX* 3.0 / 4.0, ymin = 0.0;
		double h_theta = 2.0*Pi / NPHI;
		cuDoubleComplex complex_zero = make_cuDoubleComplex(0.0, 0.0);
		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = 0; m < NPHI; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;


				cuDoubleComplex trV_V = make_cuDoubleComplex(1.0, 0.0);
				

				double r_z = 1.0 / 2.0*log(exp(2.0*x_1[j * N + i]) + exp(2.0*x_1[m * N + n])
					- 2.0*exp(x_1[j * N + i] + x_1[m * N + n])*cos(y_1[j * N + i] - y_1[m * N + n]));
				double r_z2 = exp(2.0*x_1[j * N + i]) + exp(2.0*x_1[m * N + n])
					- 2.0*exp(x_1[j * N + i] + x_1[m * N + n])*cos(y_1[j * N + i] - y_1[m * N + n]);
				if (r_z < xmin || r_z > xmax - h) {
					//trV= - S(r)
					trV_V = cuCsub(trV_V,
						S_matrix[j * N + i]);
				}
				else {
					//trV=S(r-z)
					//trV_V = cuCadd(trV_V,
					//	linear_interpolation_Smatrix(S_matrix, x_1, y_1, 2.0,
					//		Pi/4.0)
					//); 
					trV_V = cuCadd(trV_V,
						linear_interpolation_Smatrix_test(S_matrix, x_1, y_1, r_z,
							acos(30.0 ))
					);
					//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
					trV_V = cuCmul(trV_V,
						cuConj(S_matrix[m * N + n]));
					//trV=S(r-z)*S(-z) - S(r)
					trV_V = cuCsub(trV_V,
						S_matrix[j * N + i]);
				}

				if ((x_1[j * N + i] - x_1[m * N + n])*(x_1[j * N + i] - x_1[m * N + n]) < 1.0e-10 &&
					(y_1[j * N + i] - y_1[m * N + n])*(y_1[j * N + i] - y_1[m * N + n]) < 1.0e-10 || r_z2 < 0.0) {
					trV_V = make_cuDoubleComplex(
						0.0
						,
						0.0
					);
				}


				cuDoubleComplex coeff = make_cuDoubleComplex(
					simpson1*simpson2
					*exp(2.0*x_1[j*N + i] - x_1[m * N + n])/r_z2
					,
					0.0
				);

				if ((x_1[j * N + i] - x_1[m * N + n])*(x_1[j * N + i] - x_1[m * N + n]) < 1.0e-10 &&
					(y_1[j * N + i] - y_1[m * N + n])*(y_1[j * N + i] - y_1[m * N + n]) < 1.0e-10) {
					coeff = make_cuDoubleComplex(
						0.0
						,
						0.0
					);
				}
				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));

			}
		}
		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h_theta*ALPHA_S_BAR / 2.0 / Pi, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
		
	}
}


void Integration_in_BK_equation(std::complex<double>* Smatrix_in, std::complex<double>* Integrated_out)
{


	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h * NX / 2.0, xmin = -h * NX / 2.0, ymin = -h * NX / 2.0;
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


void Integration_in_logscale_BK_equation(std::complex<double>* Smatrix_in, std::complex<double>* Integrated_out)
{

	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX* 3.0 / 4.0, ymin = 0.0;
	double   *x = new double[N*NPHI], *y = new double[N*NPHI];
	for (int j = 0; j < NPHI; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i * h;
			y[NX*j + i] = ymin + j * h_theta;
		}
	}

	// Allocate arrays on the device
	double  *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*NPHI);
	cudaMalloc((void**)&y_d, sizeof(double)*N*NPHI);
	cudaMemcpy(x_d, x, sizeof(double)*N*NPHI, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*NPHI, cudaMemcpyHostToDevice);

	cuDoubleComplex *S_matrix_d;
	cudaMalloc((void**)&S_matrix_d, sizeof(cuDoubleComplex) * N*NPHI);
	cudaMemcpy(S_matrix_d, Smatrix_in, sizeof(std::complex<double>) * N*NPHI, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*NPHI);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((NPHI - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	//Integration_BK_logscale_direct_test <<<dimGrid, dimBlock >>> (Integrated_d, S_matrix_d, x_d, y_d, h, N);
	Integration_BK_logscale_direct <<<dimGrid, dimBlock >>> (Integrated_d, S_matrix_d, x_d, y_d, h, N);
	//test_dayo <<<dimGrid, dimBlock >>> ( x_d, y_d);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
		std::cout << "close the window" << '\n';
		exit(-1);
	}

	cudaMemcpy(Integrated_out, Integrated_d, sizeof(std::complex<double>)*N*NPHI, cudaMemcpyDeviceToHost);


	cudaFree(x_d);
	cudaFree(y_d);;
	cudaFree(Integrated_d);
	cudaFree(S_matrix_d);
	delete[](x);
	delete[](y);
}
