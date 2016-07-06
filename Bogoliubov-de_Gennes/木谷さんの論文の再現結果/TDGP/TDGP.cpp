#pragma comment(lib, "libfftw3-3.lib")

#define _USE_MATH_DEFINES
#include <fftw3.h>
#include <cmath>
#include<stdio.h>
#include<iostream>
#include<cstdio>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <complex>

using Complex = std::complex<double>;
inline fftw_complex* fftwcast(Complex* f){ return reinterpret_cast<fftw_complex*>(f); }


using namespace std;

const int N = 256;/*��������*/

/*GP�Œ������o��*/

double gN = 3.72; /*���ݍ�p�̋���*/

double int_psi(double x){ return x*exp(-0.5*x*x); } /*�����֐�*/

double pot(double x){ return 25 * pow(sin(x / 5), 2); } /*�ߑ��|�e���V����*/
double ppot(double x){ return 25 * pow(sin(x/5-0.01/5),2); }/*�ۓ��|�e���V����*/

double L = 10.0; /*�n�̒���*/
double Time = 7.0; /*���W�����鎞��*/

double h = L / (N+1); /*���������̔�����*/
double k = 0.01; /*���������̔�����*/

double simpson(double temp[N]){
	double s; /*Simpson�ϕ��Ńm�����̓������߂邽�߂Ɋe�z����悷��*/
	double sum1 = 0.0; /*4�{�����*/
	double sum2 = 0.0; /*2�{�����*/

	for (int i = 1; i <= N-2;i=i+2){
		sum1 += temp[i]*temp[i];
		sum2 += temp[i + 1]*temp[i+1];
	}
	s = (h/3.0)*(temp[0]*temp[0] + 4 * sum1 + 2 * sum2 + temp[N-1]*temp[N-1]);
	return s;
}


double kikaku(double x, double psinorm){return x/(sqrt(psinorm)) ; }/*�K�i��(�m�����Ŋ���)*/

double make_a(double x,double mu,double V){return -2+h*h*(mu-V-gN*pow(fabs(x),2)-(2/k)) ; }

double make_b(double x2, double x1, double x0, double mu,double V){ return -x2 - (-2 + h*h*(mu - V - gN*pow(fabs(x1), 2) + (2 / k)))*x1-x0; }


/*TDGP*/
double i2k(int m){
	if (m < N / 2){ return m / (double)L; }
	else{ return (m - N) / (double)L; }
}


int main(void){


	//double* fi = new double[N];
	/*GP*/
	double* psi0 = new double[N];/*�����z���(x,0),���̌��GaussSeidel�@�̏����l���Ƃ��Ďg�p*/
	double* psi_t = new double[N];/*�K�i�������z���(x,t)*/
	double psinorm;

	double mu;

	double a[N][N] = {};
	double* b= new double[N]();

	/*TDGP*/
	Complex *f_in = new Complex[N];
	//Complex *f_out = new Complex[N];
	Complex *F_in = new Complex[N];
	//Complex *psi = new Complex[N];


	/*GP�Œ������o��*/
	for (int i = 0; i < N; i++){ /*�����z���(x,0)�̍쐬*/
		psi0[i] = int_psi((-L/2)+i*h);
	}
	
	/*for (int i = 0; i < N; i++){
		printf("x=%f : %f \n", (-L/2)+i*h, psi0[i]);
	}*/

	psinorm = simpson(psi0);

	//printf("norm_int = %f \n",norm);

	for (int i = 0; i < N; i++){ /*�z��Ղ̋K�i��*/
		psi_t[i] = kikaku(psi0[i], psinorm);
		psi0[i] = 0; /*��(x,t+1)�̏����z��*/
	}


	/*for (int i = 0; i < N; i++){
		printf("psi_t[%d] = %f \n", i, psi_t[i]);
	}*/

	mu = 3;/*�ʂ̏����l��ݒ�*/

	for (int t = 0; k*t < Time; t++){ /*���Ԕ��W*/

		/*stringstream FileName;
		FileName << "GP" << t << ".txt";
		ofstream ofs(FileName.str().c_str());
		for (int i = 0; i < N; i++){
			ofs << (-L / 2) + i*h << "\t" << psi_t[i] * psi_t[i] << endl;
		}*/

		


		/*�s��a�i��2.25�j�̍��̂�ƁA�x�N�g�����i2.25�̉E�̂�j*/
		a[0][0] = 1.0;
		a[0][1] = 0;
		b[0] = 0;
		for (int i = 1; i <= N - 2; i++){
			a[i][i - 1] = 1.0;
			a[i][i] = make_a(psi_t[i], mu, pot((-L / 2) + i*h));
			a[i][i + 1] = 1.0;
			b[i] = make_b(psi_t[i + 1], psi_t[i], psi_t[i - 1], mu, pot((-L / 2) + i*h));
		}
		a[N - 1][N - 2] = 0;
		a[N - 1][N - 1] = 1.0;
		b[N - 1] = 0;
		/**/



		/*GaussSeidel�@�Ŏ��Ԃ���i�񂾃Ղ����߂�Bpsi0�Ɋi�[(���E�����ɂ��psi0[0]=psi0[nx-1]=0)*/
		psi0[0] = 0.0;
		psi0[N - 1] = 0.0;
		
		int j = 0;/*�J�E���^*/
		
		while (j<500){
			
			j += 1;
			for (int i = 1; i <= N - 2; i++){
				psi0[i] = (1 / a[i][i])*(b[i] - a[i][i - 1] * psi0[i - 1] - a[i][i+1] * psi0[i + 1]);
			}
		}/*��(x,t+1)��GaussSeidel�@�Ŏ����������������Ă�͂�*/
		/**/

		/*�Ղ̋K�i��*/
		psinorm = simpson(psi0);
		for (int i = 0; i < N; i++){ /*�z��Ղ̋K�i��*/
			psi_t[i] = kikaku(psi0[i], psinorm);
			psi0[i] = 0;
		}

		/*�ʂ�␳*/
		mu = mu - (1 / (2 * k))*(psinorm - 1);
	}

	
	stringstream FileName;
	FileName << "GP.txt";
	ofstream ofs(FileName.str().c_str());
	for (int i = 0; i < N; i++){ //�ex���Ƃ̃Փ����o��
		ofs << (-L / 2) + i*h << "\t" << psi_t[i] * psi_t[i] << endl;
	}
	

	
	
	/*TDGP*/
	
	

	
	double dt = 1E-4;
	fftw_plan plan1 = fftw_plan_dft_1d(N, fftwcast(f_in), fftwcast(f_in), FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan plan = fftw_plan_dft_1d(N, fftwcast(F_in), fftwcast(f_in), FFTW_BACKWARD, FFTW_MEASURE);

	for (int i = 0; i < N; i++){
		f_in[i] = (Complex)psi_t[i];
	}

	for (int t = 0; t <= 900000;t++){
		
		if (true){
			for (int i = 0; i < N ; i++){
				//f_in[i] = f_in[i] * exp(Complex(0,-(-mu+gN*norm(f_in[i]))*dt));
				f_in[i] = f_in[i] * exp(Complex(0, -(ppot((-L/2)+i*h) - mu + gN*norm(f_in[i]))*dt));
			}
		}
		else{
			for (int i = 0; i < N ; i++){
				//f_in[i] = psi[i] * Complex(cos(t*k*(pot((-L / 2) + i*h) - mu + gN*(pow(real(psi[i]), 2) + pow(imag(psi[i]), 2)))), -1 * sin(t*k*(pot((-L / 2) + i*h) - mu + gN*(pow(real(psi[i]), 2) + pow(imag(psi[i]), 2)))));
				f_in[i] = f_in[i] * exp(-1 * k*t*(pot((-L / 2) + i*h) - mu + gN*(pow(real(f_in[i]), 2) + pow(imag(f_in[i]), 2)))*sqrt(Complex(-1, 0)));
			}
		}

		fftw_execute(plan1);


		
		for (int i = 0; i< N; i++){
			//F_in[i] = f_in[i];
			//F_in[m] = f_in[m] * exp(-1*dt*(pow(2 * M_PI*i2k(m), 2))*Complex(0, 1));
			F_in[i] = f_in[i] * exp(Complex(0,-(pow(2*M_PI*i2k(i),2))*dt))/(N*1.0);
			//F_in[m] = f_in[m] * Complex(cos(t*k*(pow(2 * M_PI*i2k(m), 2))), -1 * sin(t*k*(pow(2 * M_PI*i2k(m), 2))));
		}
		fftw_execute(plan);

		
		/*for (int i = 0; i < N;i++){
			f_out[i] = f_out[i] / (N*1.0);
		}*/

		if (t%1000==0){
			stringstream FileName;

			FileName << "TDGP" << t << ".txt";

			ofstream ofs(FileName.str().c_str());
			for (int i = 0; i < N; i++){
				ofs << setw(10) << (-L / 2) + i*h << "\t" << norm(f_in[i]) << endl;

			}
		}
		/*
		for (int i = 0; i < N;i++){
			psi[i] = f_out[i];
		}*/
	}

	fftw_destroy_plan(plan1);
	fftw_destroy_plan(plan);


	delete[] psi0;
	delete[] b;


	delete[] F_in;
	//delete[] f_out;
	delete[] f_in;
	//delete[] psi;

	delete[] psi_t;
	

	return 0;
}