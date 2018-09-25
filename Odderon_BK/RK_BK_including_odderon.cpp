// This program is intended to solve the BK equation
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
#include <functional>
#include <vector>
#include <cctype>
#include <complex>
#include <math.h>
#include <stdio.h>
#include <random>

#include "Parameters.h"
#include "Spline.h"
#include "INTDE2_functional.h"
#include "linear_interpolation.h"

using namespace std;

//#define DEBUG

//#define HIGH_PRECISION

//not to compare Berger Stasto
#define SOL_IN_D

#define RUNGEKUTTA

#define LARGE_Nc

#define LOGSCALE

//for DE formula
int nfunc;

#ifdef LARGE_Nc
const double exp_coeff_U = 3.0 / 2.0;
#else
const double exp_coeff_U = 4.0 / 3.0;
#endif

/**
* The size of nuclear
*/
const double Radius_Nuc = 1.0;


#define	  	Nc		  	3


/**
* The impact parameter
*/
const double IMPACTP_B = 1.0;

/**
* The evolution step size.
*/
#define DELTA_T         0.1
#define OUTPUT_DELTA_T  5.0
#define END_T           25.0


/**
* The evolution rapidity. tau = bar{alpha}_s * rapidity
*/
double tau = 0;
double inter_tau = 0.5;

const double initialQ0 = 1.0;
const double initial_C = 1.0;


void init_BK(std::complex<double>* Smatrix_in){
	//tau = 0;
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h * NX / 1.0, xmin = -h * NX / 2.0, ymin = -h * NX / 2.0;
	double   *x = new double[N*N], *y = new double[N*N];
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i * h;
			y[NX*j + i] = ymin + j * h;
			Smatrix_in[NX*j + i] = complex<double>(
				exp(-(x[NX*j + i]* x[NX*j + i]+ y[NX*j + i]* y[NX*j + i])*initialQ0*initialQ0),
				exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i])*initialQ0*initialQ0)
				*(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i])*y[NX*j + i]
				*initial_C
				);
		}
	}


}


void init_BK_log(std::complex<double>* Smatrix_in) {
	//tau = 0;
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX *3.0/ 4.0, ymin = 0.0;
	double   *x = new double[N*N], *y = new double[N*N];
	for (int j = 0; j < NPHI; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i * h;
			y[NX*j + i] = ymin + j * h_theta;
			Smatrix_in[NX*j + i] = complex<double>(
				exp(-exp(2.0*x[NX*j + i])*initialQ0*initialQ0),
				exp(-exp(2.0*x[NX*j + i] )*initialQ0*initialQ0)
				*exp(3.0*x[NX*j + i] )*sin(y[NX*j + i])
				* initial_C
				);
		}
	}


}


//S=S0+iQ = 1-N -> N = 1-S0 + iQ'
void init_BK_log_Ncalculation(std::complex<double>* Smatrix_in) {
	//tau = 0;
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX *3.0 / 4.0, ymin = 0.0;
	double   *x = new double[N*N], *y = new double[N*N];
	for (int j = 0; j < NPHI; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i * h;
			y[NX*j + i] = ymin + j * h_theta;
			Smatrix_in[NX*j + i] = complex<double>(
				1.0-exp(-exp(2.0*x[NX*j + i])*initialQ0*initialQ0),
				exp(-exp(2.0*x[NX*j + i])*initialQ0*initialQ0)
				*exp(3.0*x[NX*j + i])*sin(y[NX*j + i])
				* initial_C
				);
		}
	}


}



//S=S0+iQ = 1-N -> N = 1-S0 + iQ'
void init_BK_log_Ncalculation_one(std::complex<double>* Smatrix_in) {
	//tau = 0;
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX *3.0 / 4.0, ymin = 0.0;
	double   *x = new double[N*N], *y = new double[N*N];
	for (int j = 0; j < NPHI; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i * h;
			y[NX*j + i] = ymin + j * h_theta;
			Smatrix_in[NX*j + i] = complex<double>(
				exp(-exp(2.0*x[NX*j + i])*initialQ0*initialQ0),
				0.0
				);
		}
	}


}

//input S_matrix NX*NPHI, x_1 NX, p_1 PHI;
std::complex<double> linear_interpolation_Smatrix_cpp(std::complex<double>* S_matrix,
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

	std::complex<double> interpolate;

	interpolate = std::complex<double>(
		(1.0 - t)*(1.0 - u)*S_matrix[i_p*NX + i_x].real() + t * (1.0 - u)*S_matrix[i_p*NX + i_x + 1].real()
		+ t * u*S_matrix[(i_p + 1)*NX + i_x + 1].real() + (1.0 - t)*u*S_matrix[(i_p + 1)*NX + i_x].real(),
		(1.0 - t)*(1.0 - u)*S_matrix[i_p*NX + i_x].imag() + t * (1.0 - u)*S_matrix[i_p*NX + i_x + 1].imag()
		+ t * u*S_matrix[(i_p + 1)*NX + i_x + 1].imag() + (1.0 - t)*u*S_matrix[(i_p + 1)*NX + i_x].imag()
		);

	return interpolate;
}


void Integration_BK_direct_cpp(std::complex<double>* integrated, std::complex<double>* S_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N) {


#pragma omp parallel for num_threads(6)
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = j * N + i;
			integrated[index] = std::complex<double>(0.0, 0.0);
			//sit the index which is center of the gaussian.

			std::complex<double> complex_zero = std::complex<double>(0.0, 0.0);
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

					std::complex<double> trV_V = std::complex<double>(0.0, 0.0);
					//if r-z is out of the region then we take the S(r-z) =0.
					if ((j - m + N / 2) < 0 || (j - m + N / 2) > N - 1 || (i - n + N / 2) < 0 || (i - n + N / 2) > N - 1) {
						//trV= - S(r)
						trV_V = trV_V
							- S_matrix[j * N + i];
					}
					else {
						//trV=S(r-z)
						trV_V = trV_V
							+ S_matrix[(j - m + N / 2) * N + i - n + N / 2];
						//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
						trV_V = trV_V
							* std::conj(S_matrix[m * N + n]);
						//trV=S(r-z)*S(-z) - S(r)
						trV_V = trV_V
							- S_matrix[j * N + i];
					}

					std::complex<double> coeff = std::complex<double>(
						simpson1*simpson2
						*(x_1[j*N + i] * x_1[j*N + i] + y_1[j*N + i] * y_1[j*N + i])
						/ ((x_1[m*N + n] - x_1[j*N + i])*(x_1[m*N + n] - x_1[j*N + i]) + (y_1[m*N + n] - y_1[j*N + i])*(y_1[m*N + n] - y_1[j*N + i]))
						/ (x_1[m*N + n] * x_1[m*N + n] + y_1[m*N + n] * y_1[m*N + n]),
						0.0
						);

					if (((x_1[m*N + n] - x_1[j*N + i])*(x_1[m*N + n] - x_1[j*N + i])
						+ (y_1[m*N + n] - y_1[j*N + i])*(y_1[m*N + n] - y_1[j*N + i])) < 1.0e-12
						|| (x_1[m*N + n] * x_1[m*N + n] + y_1[m*N + n] * y_1[m*N + n]) < 1.0e-12) {
						coeff = std::complex<double>(0.0, 0.0);
					}

					integrated[index] = integrated[index] + coeff * trV_V;

				}
			}

			std::complex<double> coeff2 = std::complex<double>(h*h*ALPHA_S_BAR / 2.0 / Pi, 0.0);

			integrated[index] = integrated[index] * coeff2;
		}
	}
}


void Integration_BK_logscale_direct_cpp(std::complex<double>* integrated, std::complex<double>* S_matrix,
	double* x_1, double* y_1, double h, int N) {

#pragma omp parallel for num_threads(6)
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < NPHI; j++) {
			int index = j * N + i;
			integrated[index] = std::complex<double>(0.0, 0.0);
			//sit the index which is center of the gaussian.
			

			std::complex<double> complex_zero = std::complex<double>(0.0, 0.0);
			double   xmax = h * NX / 4.0, xmin = -h * NX* 3.0 / 4.0, ymin = 0.0;
			double h_theta = 2.0*Pi / NPHI;
			//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
			for (int m = 0; m < NPHI; m++) {
				for (int n = 0; n < N; n++) {
					double simpson1 = 1.0;
					double simpson2 = 1.0;


					if (n == 0 || n == N - 1) {
						simpson2 = 1.0 / 3.0;
					}
					else if (n % 2 == 0) {
						simpson2 = 2.0 / 3.0;
					}
					else {

						simpson2 = 4.0 / 3.0;
					}

					std::complex<double> trV_V = std::complex<double>(0.0, 0.0);

					double r_z = 1.0 / 2.0*log(exp(2.0*x_1[j * N + i]) + exp(2.0*x_1[m * N + n])
						- 2.0*exp(x_1[j * N + i] + x_1[m * N + n])*cos(y_1[j * N + i] - y_1[m * N + n]));
					double r_z2 = exp(2.0*x_1[j * N + i]) + exp(2.0*x_1[m * N + n])
						- 2.0*exp(x_1[j * N + i] + x_1[m * N + n])*cos(y_1[j * N + i] - y_1[m * N + n]);
					double r_z2_o_x2 = 1.0 + exp(2.0*x_1[m * N + n] - 2.0*x_1[j * N + i])
						- 2.0*exp(-x_1[j * N + i] + x_1[m * N + n])*cos(y_1[j * N + i] - y_1[m * N + n]);
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
					if (r_z < xmin) {
						std::complex<double> unit = std::complex<double>(1.0, 0.0);
						//trV=S(r-z)
						//trV_V += unit;
						//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
						//trV_V = trV_V
						//	* std::conj(S_matrix[m * N + n]);
						//trV=S(r-z)*S(-z) - S(r)
						//trV_V = trV_V
						//	- S_matrix[j * N + i];

						//trV=S(r-z)*S(-z) - S(r)
						trV_V = unit * std::conj(S_matrix[m * N + n]) - S_matrix[j * N + i];
					}
					else if (r_z > xmax - h) {

						std::complex<double> zero = std::complex<double>(0.0, 0.0);
						//trV=S(r-z)
						//trV_V += zero;
						//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
						//trV_V = trV_V
						//	* std::conj(S_matrix[m * N + n]);
						//trV=S(r-z)*S(-z) - S(r)
						//trV_V = trV_V
						//	- S_matrix[j * N + i];

						//trV=S(r-z)*S(-z) - S(r)
						trV_V = zero * std::conj(S_matrix[m * N + n]) - S_matrix[j * N + i];
					}
					else {
						//trV=S(r-z)
						//trV_V = trV_V
						//	+ linear_interpolation_Smatrix_cpp(S_matrix, x_1, y_1, r_z,
						//		acos(angletocos));
						//trV=S(r-z)*S(-z) <- S(-x) = S(x)^*
						//trV_V = trV_V
						//	* std::conj(S_matrix[m * N + n]);
						//trV=S(r-z)*S(-z) - S(r)
						//trV_V = trV_V
						//	- S_matrix[j * N + i];

						//trV=S(r-z)*S(-z) - S(r)
						trV_V = linear_interpolation_Smatrix_cpp(S_matrix, x_1, y_1, r_z,
							acos(angletocos))
							* std::conj(S_matrix[m * N + n]) - S_matrix[j * N + i];

					}//Caution!!! nan * 0 = nan
					if (((x_1[j * N + i] - x_1[m * N + n])*(x_1[j * N + i] - x_1[m * N + n]) < 1.0e-10 &&
						(y_1[j * N + i] - y_1[m * N + n])*(y_1[j * N + i] - y_1[m * N + n]) < 1.0e-10) || r_z2 < 0.0) {
						trV_V = std::complex<double>(
							0.0
							,
							0.0
							);
					}

					std::complex<double> coeff = std::complex<double>(
						simpson1*simpson2
						*1.0
						/ r_z2_o_x2,
						0.0
						);

					if ( ((x_1[j * N + i] - x_1[m * N + n])*(x_1[j * N + i] - x_1[m * N + n]) < 1.0e-10 &&
						(y_1[j * N + i] - y_1[m * N + n])*(y_1[j * N + i] - y_1[m * N + n]) < 1.0e-10 ) || 
						((x_1[j * N + i] - x_1[m * N + n])*(x_1[j * N + i] - x_1[m * N + n]) < 1.0e-10 &&
						(y_1[j * N + i] - y_1[m * N + n] + 2.0*Pi)*(y_1[j * N + i] - y_1[m * N + n] + 2.0*Pi) < 1.0e-10)
						) {
						coeff = std::complex<double>(0.0, 0.0);
					}

					if(coeff != coeff){
						cout << ((x_1[m*N + n] - x_1[j*N + i])*(x_1[m*N + n] - x_1[j*N + i])
							+ (y_1[m*N + n] - y_1[j*N + i])*(y_1[m*N + n] - y_1[j*N + i]))
							<< "\t" << (x_1[m*N + n] * x_1[m*N + n] + y_1[m*N + n] * y_1[m*N + n]) << "\n";
					}
					if (trV_V != trV_V) {
						cout << index << "\t" << m * N + n << "\n";
					}
					integrated[index] = integrated[index] + coeff * trV_V;

					if (integrated[index] != integrated[index]) {
						cout << "internal \t trV_V " << trV_V <<"\t coeff "<<coeff <<"\t index "  << index << "\n";
					}

				}
			}

			std::complex<double> coeff2 = std::complex<double>(h*h_theta*ALPHA_S_BAR / 2.0 / Pi, 0.0);
		

			if (integrated[index] != integrated[index]) {
				cout << index<< "\n";
			}
			integrated[index] = integrated[index] * coeff2;
		}
	}
}


void Integration_in_BK_equation_cpp(std::complex<double>* Smatrix_in, std::complex<double>* Integrated_out)
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


	Integration_BK_direct_cpp(Integrated_out, Smatrix_in, x, y, h, 0, N, N);



	delete[](x);
	delete[](y);
}


void Integration_in_logscale_BK_equation_cpp(std::complex<double>* Smatrix_in, std::complex<double>* Integrated_out)
{


	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX* 3.0 / 4.0, ymin = 0.0;
	double   *x = new double[N*N], *y = new double[N*N];
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i * h;
			y[NX*j + i] = ymin + j * h_theta;
		}
	}


	Integration_BK_logscale_direct_cpp(Integrated_out, Smatrix_in, x, y, h, N);



	delete[](x);
	delete[](y);
}



void Integration_in_BK_equation(std::complex<double>* Smatrix_in, std::complex<double>* Integrated_out);


void f_one_step_BK_complex(
	vector<complex<double>> &sol_BMS_g_in,
	vector<complex<double>> &NEW_BMS_g_in, const double dtau)
{
	vector<complex<double>> temp_s_BMS_in(NX*NX, 0);

#pragma omp parallel for num_threads(6)
	for (int veint = 0; veint < NX*NX; veint++) {
		temp_s_BMS_in[veint] = sol_BMS_g_in[veint];
	}

	Integration_in_BK_equation(temp_s_BMS_in.data(), NEW_BMS_g_in.data());
	//Integration_in_BK_equation_cpp(temp_s_BMS_in.data(), NEW_BMS_g_in.data());

}

void s_one_step_BK_complex(const vector<complex<double>> &K1_g_in, 
	vector<complex<double>> &sol_BMS_g_in, 
	vector<complex<double>> &NEW_BMS_g_in, const double dtau)
{
	vector<complex<double>> temp_s_BMS_in(NX*NX, 0);

#pragma omp parallel for num_threads(6)
	for (int veint = 0; veint < NX*NX; veint++) {
		temp_s_BMS_in[veint] = sol_BMS_g_in[veint] + dtau / 2.0*K1_g_in[veint];
	}

	Integration_in_BK_equation(temp_s_BMS_in.data(), NEW_BMS_g_in.data());
	//Integration_in_BK_equation_cpp(temp_s_BMS_in.data(), NEW_BMS_g_in.data());

}

void one_step_complex(vector<complex<double>> &sol_BK, double dtau){


	vector<complex<double>> K1_g_comp(NX*NX, 0);
	vector<complex<double>> K2_g_comp(NX*NX, 0);
	vector<complex<double>> K3_g_comp(NX*NX, 0);
	vector<complex<double>> K4_g_comp(NX*NX, 0);

	//evolve using the BK equation
	 f_one_step_BK_complex(sol_BK, K1_g_comp, dtau);
	 s_one_step_BK_complex(K1_g_comp, sol_BK, K2_g_comp, dtau);
	 s_one_step_BK_complex(K2_g_comp, sol_BK, K3_g_comp, dtau);
	 s_one_step_BK_complex(K3_g_comp, sol_BK, K4_g_comp, 2.0*dtau);

	tau += dtau; 
	vector<complex<double>> Old_BK_in(NX*NX,0);
	Old_BK_in = sol_BK;

#pragma omp parallel for num_threads(6)
	for (int i = 0; i < NX*NX; ++i) {
		sol_BK[i] = Old_BK_in[i]
			+ dtau / 6.0*(K1_g_comp[i] + 2.0*K2_g_comp[i] + 2.0*K3_g_comp[i] + K4_g_comp[i]);
	}
}

void Integration_in_logscale_BK_equation(std::complex<double>* Smatrix_in, std::complex<double>* Integrated_out);

void f_one_step_logscale_BK_complex(
	vector<complex<double>> &sol_BMS_g_in,
	vector<complex<double>> &NEW_BMS_g_in, const double dtau)
{
	vector<complex<double>> temp_s_BMS_in(NX*NPHI, 0);

#pragma omp parallel for num_threads(6)
	for (int veint = 0; veint < NX*NPHI; veint++) {
		temp_s_BMS_in[veint] = sol_BMS_g_in[veint];
	}

	//Integration_in_logscale_BK_equation(temp_s_BMS_in.data(), NEW_BMS_g_in.data());
	Integration_in_logscale_BK_equation_cpp(temp_s_BMS_in.data(), NEW_BMS_g_in.data());

}

void s_one_step_logscale_BK_complex(const vector<complex<double>> &K1_g_in,
	vector<complex<double>> &sol_BMS_g_in,
	vector<complex<double>> &NEW_BMS_g_in, const double dtau)
{
	vector<complex<double>> temp_s_BMS_in(NX*NPHI, 0);

#pragma omp parallel for num_threads(6)
	for (int veint = 0; veint < NX*NPHI; veint++) {
		temp_s_BMS_in[veint] = sol_BMS_g_in[veint] + dtau / 2.0*K1_g_in[veint];
	}

	//Integration_in_logscale_BK_equation(temp_s_BMS_in.data(), NEW_BMS_g_in.data());
	Integration_in_logscale_BK_equation_cpp(temp_s_BMS_in.data(), NEW_BMS_g_in.data());

}

void one_step_logscale_complex(vector<complex<double>> &sol_BK, double dtau) {


	vector<complex<double>> K1_g_comp(NX*NPHI, 0);
	vector<complex<double>> K2_g_comp(NX*NPHI, 0);
	vector<complex<double>> K3_g_comp(NX*NPHI, 0);
	vector<complex<double>> K4_g_comp(NX*NPHI, 0);

	//evolve using the BK equation
	f_one_step_logscale_BK_complex(sol_BK, K1_g_comp, dtau);
	s_one_step_logscale_BK_complex(K1_g_comp, sol_BK, K2_g_comp, dtau);
	s_one_step_logscale_BK_complex(K2_g_comp, sol_BK, K3_g_comp, dtau);
	s_one_step_logscale_BK_complex(K3_g_comp, sol_BK, K4_g_comp, 2.0*dtau);

	tau += dtau;
	vector<complex<double>> Old_BK_in(NX*NPHI, 0);
	Old_BK_in = sol_BK;

#pragma omp parallel for num_threads(6)
	for (int i = 0; i < NX*NPHI; ++i) {
		sol_BK[i] = Old_BK_in[i]
			+ dtau / 6.0*(K1_g_comp[i] + 2.0*K2_g_comp[i] + 2.0*K3_g_comp[i] + K4_g_comp[i]);
	}
}

/**
* Prints g_Y.
*/
void print_g(vector<complex<double>> &sol_BK) {
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

	ostringstream ofilename,ofilename2;
	ofilename << "G:\\hagiyoshi\\Data\\BK_odderon\\solutions\\BK_res_size" 
		<< LATTICE_SIZE << "_grid_" << NX << "_timestep_" << DELTA_T << "_t_" << tau << "_hipre.txt";
	ofstream ofs_res(ofilename.str().c_str());
	ofilename2 << "G:\\hagiyoshi\\Data\\BK_odderon\\solutions\\BK_res_distance_size"
		<< LATTICE_SIZE << "_grid_" << NX << "_timestep_" << DELTA_T << "_t_" << tau << "_hipre.txt";
	ofstream ofs_res2(ofilename2.str().c_str());

	ofs_res << "# x \t y \t Re( S ) \t Im( S )"<< endl;
	ofs_res2 << "# r \t Re( S ) \t Im( S )" << endl;

	for (int i = 0; i <NX; i++){
		for (int j = 0; j < NX; j++) {
			ofs_res  << scientific << x[NX*j + i] << "\t" << y[NX*j + i] << "\t"
				<< sol_BK[NX*j + i].real() << "\t" << sol_BK[NX*j + i].imag() << endl;
		}

		ofs_res << endl;
	}

	for (int i = N / 2; i < N; i++) {

		ofs_res2 << scientific << sqrt(x[NX*i + i] * x[NX*i + i] + y[NX*i + i] * y[NX*i + i]) << "\t"
			<< sol_BK[NX*i + i].real() << "\t" << sol_BK[NX*i + i].imag() << endl;
	}

	delete[](x);
	delete[](y);
}


/**
* Prints g_Y.
*/
void print_logscale_g(vector<complex<double>> &sol_BK) {
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double h_theta = 2.0*Pi / NPHI;
	double   xmax = h * NX / 4.0, xmin = -h * NX*3.0 / 4.0, ymin = 0.0;
	double   *x = new double[N*NPHI], *y = new double[N*NPHI];
	for (int j = 0; j < NPHI; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i * h;
			y[NX*j + i] = ymin + j * h_theta;
		}
	}

	ostringstream ofilename, ofilename2;
	ofilename << "G:\\hagiyoshi\\Data\\BK_odderon\\solutions\\BK_logscale_res_size"
		<< LATTICE_SIZE << "_grid_" << NX << "_phi_" << NPHI << "_timestep_" << DELTA_T << "_t_" << tau << "_hipre.txt";
	ofstream ofs_res(ofilename.str().c_str());
	ofilename2 << "G:\\hagiyoshi\\Data\\BK_odderon\\solutions\\BK_logscale_res_distance_size"
		<< LATTICE_SIZE << "_grid_" << NX << "_phi_" << NPHI << "_timestep_" << DELTA_T << "_t_" << tau << "_hipre.txt";
	ofstream ofs_res2(ofilename2.str().c_str());

	ofs_res << "# r \t phi \t Re( S ) \t Im( S )" << endl;
	ofs_res2 << "# r pi/2 \t Re( S ) \t Im( S )" << endl;

	for (int i = 0; i < NPHI; i++) {
		for (int j = 0; j < NX; j++) {
			ofs_res << scientific << exp(x[NX*i + j]) << "\t" << y[NX*i + j] << "\t"
				<< sol_BK[NX*i + j].real() << "\t" << sol_BK[NX*i + j].imag() << endl;
		}

		ofs_res << endl;
	}

	for (int i = 0; i < N; i++) {

		ofs_res2 << scientific << exp(x[NX*NPHI/4 + i]) << "\t"
			<< sol_BK[NX*NPHI / 4 + i].real() << "\t" << sol_BK[NX*NPHI / 4 + i].imag() << endl;
	}

	delete[](x);
	delete[](y);
}


int main(){
	time_t t0 = time(NULL);
	ostringstream ofilename;
#ifdef LOGSCALE
	ofilename << "G:\\hagiyoshi\\Data\\BK_odderon\\solutions\\BK_logscale_res_taudep_size"
		<< LATTICE_SIZE << "_grid_" << NX << "_phi_" << NPHI << "_timestep_" << DELTA_T << "_hipre.txt";
#else
	ofilename << "G:\\hagiyoshi\\Data\\BK_odderon\\solutions\\BK_res_taudep_size"
		<< LATTICE_SIZE << "_grid_" << NX << "_timestep_" << DELTA_T << "_hipre.txt";
#endif
	ofstream ofs_res(ofilename.str().c_str());

	ofs_res << "# tau \t r \t Re( S ) \t Im( S )" << endl;

#ifdef LOGSCALE
	vector<complex<double>> sol_BK_comp(NX*NPHI, 0);
	double h_phi = 2.0*Pi / NPHI;
	double h_half = 1.0*LATTICE_SIZE / NX;
#else
	double h_half = 1.0*LATTICE_SIZE / NX;
	vector<complex<double>> sol_BK_comp(NX*NX, 0);
#endif
   
        
	const double EPS = 1e-12;

	double next_tau = 0;

#ifdef LOGSCALE
	init_BK_log(sol_BK_comp.data());
	//init_BK_log_Ncalculation(sol_BK_comp.data());
	//init_BK_log_Ncalculation_one(sol_BK_comp.data());
	print_logscale_g(sol_BK_comp);
#else
	init_BK(sol_BK_comp.data());
	print_g(sol_BK_comp);
#endif

	//evaluate the number of time step
	for (;;) {
		int reunit_count = 0;
		if (tau >= END_T - EPS) {
			break;
		}
		next_tau = min(next_tau + OUTPUT_DELTA_T, END_T);
		while (tau < next_tau - EPS) {
#ifdef LOGSCALE
			one_step_logscale_complex(sol_BK_comp,min(DELTA_T, next_tau - tau));
#else
			one_step_complex(sol_BK_comp, min(DELTA_T, next_tau - tau));
#endif
		}
#ifdef LOGSCALE
		print_logscale_g(sol_BK_comp);
		ofs_res << scientific << tau << "\t" << exp( 3.0 * NX / 4.0*h_half) << "\t"
			<< sol_BK_comp[NPHI / 4 * NX + 3 * NX / 4].real() << "\t" << sol_BK_comp[NPHI / 4 * NX + 3 * NX / 4].imag() << "\n";
		cout << "time \t" << tau << "\n";
#else
		print_g(sol_BK_comp);
		ofs_res << scientific << tau<< "\t" << sqrt(2.0)*h_half*(double)NX/4.0 << "\t"
			<< sol_BK_comp[3*NX / 4 * NX + 3 * NX / 4].real() << "\t" << sol_BK_comp[3 * NX / 4 * NX + 3 * NX / 4].imag() << "\n";
		cout << "time \t" << tau << "\n";
#endif
	}

#ifdef LOGSCALE
	print_logscale_g(sol_BK_comp);
	ofs_res << scientific << tau << "\t" << exp(3.0 * NX / 4.0*h_half) << "\t"
		<< sol_BK_comp[NPHI / 4 * NX + 3 * NX / 4].real() << "\t" << sol_BK_comp[NPHI / 4 * NX + 3 * NX / 4].imag() << "\n";
	cout << "time \t" << tau << "\n";
#else
	print_g(sol_BK_comp);
	ofs_res << scientific << tau << "\t" << sqrt(2.0)*h_half*(double)NX / 4.0 << "\t"
		<< sol_BK_comp[3 * NX / 4 * NX + 3 * NX / 4].real() << "\t" << sol_BK_comp[3 * NX / 4 * NX + 3 * NX / 4].imag() << "\n";
	cout << "time \t" << tau << "\n";
#endif

	time_t t1 = time(NULL);
	cout << t0 - t1 << endl;
	cout<<endl;
}
