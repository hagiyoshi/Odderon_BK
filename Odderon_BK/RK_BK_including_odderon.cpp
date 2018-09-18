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

//#define LOGSCALE

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
#define DELTA_T         0.01
#define OUTPUT_DELTA_T  0.1
#define END_T           5.0


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
				*(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i])*sqrt(y[NX*j + i]* y[NX*j + i])
				*initial_C
				);
		}
	}


}


void Integration_in_BK_equation(std::complex<double>* Smatrix_in, std::complex<double>* Integrated_out);

void f_one_step_BK_complex(
	vector<complex<double>> &sol_BMS_g_in,
	vector<complex<double>> &NEW_BMS_g_in, const double dtau)
{
	vector<complex<double>> temp_s_BMS_in(NX*NX, 0),
		temp_s_K1_in(NX*NX, 0);

#pragma omp parallel for num_threads(6)
	for (int veint = 0; veint < NX*NX; veint++) {
		temp_s_BMS_in[veint] = sol_BMS_g_in[veint];
	}

	Integration_in_BK_equation(temp_s_BMS_in.data(), NEW_BMS_g_in.data());

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
	ofilename << "G:\\hagiyoshi\\Data\\BK_odderon\\solutions\\BK_res_size" << LATTICE_SIZE << "_grid_" << NX << "_timestep_" << DELTA_T << "_t_" << tau << "_hipre.txt";
	ofstream ofs_res(ofilename.str().c_str());
	ofilename2 << "G:\\hagiyoshi\\Data\\BK_odderon\\solutions\\BK_res_distance_size" << LATTICE_SIZE << "_grid_" << NX << "_timestep_" << DELTA_T << "_t_" << tau << "_hipre.txt";
	ofstream ofs_res2(ofilename2.str().c_str());

	ofs_res << "# x \t y \t Re( S ) \t Im( S )"<< endl;
	ofs_res << "# r \t Re( S ) \t Im( S )" << endl;

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



int main(){
	time_t t0 = time(NULL);


	vector<complex<double>> sol_BK_comp(NX*NX, 0);
   
        
	const double EPS = 1e-12;

	double next_tau = 0;

	init_BK(sol_BK_comp.data());

	print_g(sol_BK_comp);

	//evaluate the number of time step
	for (;;) {
		int reunit_count = 0;
		if (tau >= END_T - EPS) {
			break;
		}
		next_tau = min(next_tau + OUTPUT_DELTA_T, END_T);
		while (tau < next_tau - EPS) {
			one_step_complex(sol_BK_comp,min(DELTA_T, next_tau - tau));
		}
		print_g(sol_BK_comp);
		cout << "time \t" << tau << "\n";
	}

	print_g(sol_BK_comp);

	time_t t1 = time(NULL);
	cout << t0 - t1 << endl;
	cout<<endl;
}
