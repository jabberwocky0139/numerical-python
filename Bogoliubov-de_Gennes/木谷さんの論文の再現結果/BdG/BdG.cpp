#include<stdio.h>
#include<iostream>
#include<cmath>
#include<cstdio>
#include <fstream>
#include <sstream>
#include <iomanip>
//#include <complex.h>
extern "C"{
#include<lapacke.h>
}

using namespace std;

const int N = 400;

/*GP�Œ������o��*/

//double gN = 4.0; /*���ݍ�p�̋���*/

double int_psi(double x){ return x*exp(-0.5*x*x); } /*�����֐�*/
//double int_psi(double x){ return x*cos(x*x); } /*�����֐�*/

double pot(double x){ return 25 * pow(sin(x / 5), 2); } /*�ߑ��|�e���V����*/


int nx = N; /*��������*/

double L = 20.0; /*�n�̒���*/
double Time = 7.0; /*���W�����鎞��*/

double h = L / (nx+1); /*���������̔�����*/
double k = 0.01; /*���������̔�����*/

double simpson(double temp[N]){
  double s; /*Simpson�ϕ��Ńm�����̓������߂邽�߂Ɋe�z����悷��*/
  double sum1 = 0.0; /*4�{�����*/
  double sum2 = 0.0; /*2�{�����*/

  for (int i = 1; i <= nx-2;i=i+2){
    sum1 += temp[i]*temp[i];
    sum2 += temp[i + 1]*temp[i+1];
  }
  s = (h/3.0)*(temp[0]*temp[0] + 4 * sum1 + 2 * sum2 + temp[nx-1]*temp[nx-1]);
  return s;
}
double kikaku(double x, double norm){return x/(sqrt(norm)) ; }/*�K�i��(�m�����Ŋ���)*/

double make_a(double x,double mu,double V,double gN){return -2+h*h*(mu-V-gN*pow(fabs(x),2)-(2/k)) ; }

double make_b(double x2, double x1, double x0, double mu,double V,double gN){ return -x2 - (-2 + h*h*(mu - V - gN*pow(fabs(x1), 2) + (2 / k)))*x1-x0; }

/*BdG�Ŏg������*/

double alpha =1 / (h*h);
double beta(double V, double mu, double x,double gN){ return 2 * (1 / (h*h)) + V - mu + 2 * gN*pow(fabs(x), 2); }
double ganma(double x,double gN){ return gN*x*x; }

double inner(double temp1[2*N],double temp2[2*N]){ /*�s��`�ʓ���*/
  double s;
  double sum1 = 0.0; /*4�{�����*/
  double sum2 = 0.0; /*2�{�����*/
	
  double* u = new double[N];
  for (int i = 0; i < N;i++){
    u[i] = temp1[i] * temp2[i] - temp1[i + N] * temp2[i + N];
  }
	
  for (int i = 1; i <= N - 2; i = i + 2){
    sum1 += u[i];
    sum2 += u[i + 1];
  }
  s = (h / 3.0)*(u[0] + 4 * sum1 + 2 * sum2 + u[N - 1]);

  delete[] u;
  return s;
}


double serchmax(double temp[2*N]){ /*�z��̒�����ő�l��������*/
  double max = 0.0;
  for (int i = 0; i < 2 * N;i++){
    if (max < temp[i]){ max = temp[i]; }
  }
  return max;
}




int main(void){

  //double* Im = new double[100];

  double gN;/****************************************/

  /*GP*/
	
	
  double norm;

  double mu;

  double a[N][N] = {};
	

  /*BdG�Ŏg�p*/
  //double T[2*N][2*N] = {};
  double Inn;

  stringstream FileName;
  FileName << "BdG.txt";
  ofstream ofs(FileName.str().c_str());
	
  for (int num = 360; num < 380;num++){
    gN = 0.01*num;

    double* psi0 = new double[N];/*�����z���(x,0),���̌��GaussSeidel�@�̏����l���Ƃ��Ďg�p*/
    double* psi_t = new double[N];/*�K�i�������z���(x,t)*/
    double* b = new double[N]();


    /*GP�Œ������o��*/
    for (int i = 0; i < nx; i++){ /*�����z���(x,0)�̍쐬*/
      psi0[i] = int_psi((-L / 2) + i*h);
    }

    norm = simpson(psi0);

    for (int i = 0; i < nx; i++){ /*�z��Ղ̋K�i��*/
      psi_t[i] = kikaku(psi0[i], norm);
      psi0[i] = 0; /*��(x,t+1)�̏����z��*/
    }

    mu = 3;/*�ʂ̏����l��ݒ�*/

    for (int t = 0; k*t < Time; t++){ /*���Ԕ��W*/

      /*stringstream FileName;
	FileName << "GP" << t << ".txt";
	ofstream ofs(FileName.str().c_str());
	for (int i = 0; i < nx; i++){
	ofs << (-L / 2) + i*h << "\t" << psi_t[i] * psi_t[i] << endl;
	}*/




      /*�s��a�i��2.25�j�̍��̂�ƁA�x�N�g�����i2.25�̉E�̂�j*/
      a[0][0] = 1.0;
      a[0][1] = 0;
      b[0] = 0;
      for (int i = 1; i <= nx - 2; i++){
	a[i][i - 1] = 1.0;
	a[i][i] = make_a(psi_t[i], mu, pot((-L / 2) + i*h), gN);
	a[i][i + 1] = 1.0;
	b[i] = make_b(psi_t[i + 1], psi_t[i], psi_t[i - 1], mu, pot((-L / 2) + i*h), gN);
      }
      a[N - 1][N - 2] = 0;
      a[N - 1][N - 1] = 1.0;
      b[N - 1] = 0;
      /**/



      /*GaussSeidel�@�Ŏ��Ԃ���i�񂾃Ղ����߂�Bpsi0�Ɋi�[(���E�����ɂ��psi0[0]=psi0[nx-1]=0)*/
      psi0[0] = 0.0;
      psi0[nx - 1] = 0.0;

      int j = 0;/*�J�E���^*/

      while (j < 500){

	j += 1;
	for (int i = 1; i <= nx - 2; i++){
	  psi0[i] = (1 / a[i][i])*(b[i] - a[i][i - 1] * psi0[i - 1] - a[i][i + 1] * psi0[i + 1]);
	}
      }/*��(x,t+1)��GaussSeidel�@�Ŏ����������������Ă�͂�*/
      /**/

      /*�Ղ̋K�i��*/
      norm = simpson(psi0);
      for (int i = 0; i < nx; i++){ /*�z��Ղ̋K�i��*/
	psi_t[i] = kikaku(psi0[i], norm);
	psi0[i] = 0;
      }

      /*�ʂ�␳*/
      mu = mu - (1 / (2 * k))*(norm - 1);
      //printf("mu=%f\n",mu);
    }
    printf("mu=%f\n",mu);

    /*
      stringstream FileName;
      FileName << "GP.txt";
      ofstream ofs(FileName.str().c_str());
      for (int i = 0; i < nx; i++){ //�e�����Ƃ̃Փ����o��
      ofs << (-L / 2) + i*h << "\t" << psi_t[i] * psi_t[i] << endl;
      }*/

    delete[] psi0;
    delete[] b;

    /************/
    /*������BdG*/
    /**********/




    /*�s��T�̍쐬*/

    /*T[0][0] = beta(pot(-L/2),mu,psi_t[0],gN);
      T[0][1] = -1*alpha;
      T[N - 1][N - 2] = -1 * alpha;
      T[N - 1][N - 1] = beta(pot((-L / 2)+(N-1)*h), mu, psi_t[N-1],gN);

      T[0][N] = ganma(psi_t[0],gN);
      T[N-1][2*N-1] = ganma(psi_t[N-1],gN);

      for (int i = 1; i <= N-2;i++){
      T[i][i - 1] = -1 * alpha;
      T[i][i] = beta(pot((-L / 2) + i*h), mu, psi_t[i],gN);
      T[i][i + 1] = -1 * alpha;

      T[i][i + N] = ganma(psi_t[i],gN);

      T[i + N][i] = -1*ganma(psi_t[i],gN);

      T[i+N][i+N - 1] = alpha;
      T[i + N][i + N] = -1 * beta(pot((-L / 2) + i*h), mu, psi_t[i],gN);
      T[i+N][i+N + 1] = alpha;
      }

      T[N][0] = -1*ganma(psi_t[0],gN);
      T[2*N - 1][N - 1] = -1*ganma(psi_t[N - 1],gN);

      T[N][N] = -1 * beta(pot(-L / 2), mu, psi_t[0],gN);
      T[N][N + 1] = alpha;
      T[2 * N - 1][2 * N - 2] = alpha;
      T[2 * N - 1][2 * N - 1] = -1 * beta(pot((-L / 2) + (N - 1)*h,gN), mu, psi_t[0]);
    */

    double* T = new double[4 * N*N]();
    /*lapack�ɓn���ׂ��z��ɒ����E�E�E*/
    /*����*/


    T[0] = beta(pot(-L / 2), mu, psi_t[0], gN);
    T[2 * N] = -1 * alpha;
    T[2 * N*N - 3 * N - 1] = -1 * alpha;
    T[2 * N*N - N - 1] = beta(pot((-L / 2) + (N - 1)*h), mu, psi_t[N - 1], gN);


    /*����*/


    T[N] = -1 * ganma(psi_t[0], gN);
    T[2 * N*N - 1] = -1 * ganma(psi_t[N - 1], gN);


    /*�E��*/


    T[2 * N*N] = ganma(psi_t[0], gN);
    T[4 * N*N - N - 1] = ganma(psi_t[N - 1], gN);


    /*�E��*/


    T[2 * N*N + N] = -1 * beta(pot(-L / 2), mu, psi_t[0], gN);
    T[2 * N*N + 3 * N] = alpha;
    T[4 * N*N - 2 * N - 1] = alpha;
    T[4 * N*N - 1] = -1 * beta(pot((-L / 2) + (N - 1)*h), mu, psi_t[N - 1], gN);



    for (int i = 1; i <= N - 2; i++){
      T[(2 * N + 1)*i - 2 * N] = -1 * alpha;
      T[(2 * N + 1)*i] = beta(pot((-L / 2) + i*h), mu, psi_t[i], gN);
      T[(2 * N + 1)*i + 2 * N] = -1 * alpha;/**/

      T[(2 * N + 1)*i + N] = -1 * ganma(psi_t[i], gN);

      T[(2 * N + 1)*i + 2 * N*N] = ganma(psi_t[i], gN);

      T[(2 * N + 1)*i + 2 * N*N - N] = alpha;
      T[(2 * N + 1)*i + 2 * N*N + N] = -1 * beta(pot((-L / 2) + i*h), mu, psi_t[i], gN);
      T[(2 * N + 1)*i + 2 * N*N + 3 * N] = alpha;
    }

    /*y0�����A���𓱏o����(y0�͉��ł��邚�ɏ㏑������邽�ߍŏ�����z������Ƃ���B)*/


    double* z = new double[2 * N];
    double* y0 = new double[2 * N];

    int* ipiv = new int[2 * N]();

    for (int i = 0; i < N; i++){
      y0[i] = psi_t[i];
      y0[i + N] = -1 * psi_t[i];

      z[i] = psi_t[i];
      z[i + N] = -1 * psi_t[i];
    }

    LAPACKE_dgesv(LAPACK_COL_MAJOR, 2 * N, 1, T, 2 * N, ipiv, z, 2 * N);
    Inn = 1 / inner(y0, z);

    delete[] T;

    /*��̌v�Z��T�͏㏑������Ă�AI�����܂����̂ōs��S���\�z����BT�̑Ίp�v�f����I�������H*/
    double* S = new double[4 * N*N]();

    /*����*/
    S[0] = beta(pot(-L / 2), mu, psi_t[0], gN);
    S[2 * N] = -1 * alpha;
    S[2 * N*N - 3 * N - 1] = -1 * alpha;
    S[2 * N*N - N - 1] = beta(pot((-L / 2) + (N - 1)*h), mu, psi_t[N - 1], gN);


    /*����*/
    S[N] = -1 * ganma(psi_t[0], gN);
    S[2 * N*N - 1] = -1 * ganma(psi_t[N - 1], gN);


    /*�E��*/
    S[2 * N*N] = ganma(psi_t[0], gN);
    S[4 * N*N - N - 1] = ganma(psi_t[N - 1], gN);


    /*�E��*/
    S[2 * N*N + N] = -1 * beta(pot(-L / 2), mu, psi_t[0], gN);
    S[2 * N*N + 3 * N] = alpha;
    S[4 * N*N - 2 * N - 1] = alpha;
    S[4 * N*N - 1] = -1 * beta(pot((-L / 2) + (N - 1)*h), mu, psi_t[N - 1], gN);



    for (int i = 1; i <= N - 2; i++){
      S[(2 * N + 1)*i - 2 * N] = -1 * alpha;
      S[(2 * N + 1)*i] = beta(pot((-L / 2) + i*h), mu, psi_t[i], gN);
      S[(2 * N + 1)*i + 2 * N] = -1 * alpha;

      S[(2 * N + 1)*i + N] = -1 * ganma(psi_t[i],gN);
 
      S[(2 * N + 1)*i + 2 * N*N] = ganma(psi_t[i],gN);

      S[(2 * N + 1)*i + 2 * N*N - N] = alpha;
      S[(2 * N + 1)*i + 2 * N*N + N] = -1 * beta(pot((-L / 2) + i*h), mu, psi_t[i], gN);
      S[(2 * N + 1)*i + 2 * N*N + 3 * N] = alpha;
    }
    



    double* wr = new double[2 * N]();
    double* wi = new double[2 * N]();

    double* vr = new double[2 * N]();
    double* vl = new double[2 * N]();


    //printf("I=%f\nS[0]=%f\nS[2*N*N-N-1]=%f\nalpha=%f\nganma=%f\n\n",I, S[0], S[2 * N*N - N - 1], alpha, S[(2*N+1)*100+N]);


    LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', 2 * N, S, 2 * N, wr, wi, vl, 2 * N, vr, 2 * N);

    /*
      for (int i = 0; i < 10;i++){
      printf("Im[%d] = %f\n",i,wi[i]);
      }
    */

    //Im[num] = serchmax(wi);
    printf("gN=%f \t maxIm=%f\n",gN, serchmax(wi));
    ofs << gN << "\t" << serchmax(wi) << endl;
    delete[] z;
    delete[] ipiv;
    delete[] y0;
    delete[] S;

    delete[] wr;
    delete[] wi;
    delete[] vr;
    delete[] vl;
    delete[] psi_t;

  }
	
  /*for (int i = 0; i < 200;i++){
    printf("gN=%f \t %f",gN,Im[i]);
    }*/



  //delete[] Im;

  return 0;
}
