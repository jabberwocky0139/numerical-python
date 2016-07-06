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

/*GPで定常解を出す*/

//double gN = 4.0; /*相互作用の強さ*/

double int_psi(double x){ return x*exp(-0.5*x*x); } /*初期関数*/
//double int_psi(double x){ return x*cos(x*x); } /*初期関数*/

double pot(double x){ return 25 * pow(sin(x / 5), 2); } /*捕捉ポテンシャル*/


int nx = N; /*差分化数*/

double L = 20.0; /*系の長さ*/
double Time = 7.0; /*発展させる時間*/

double h = L / (nx+1); /*ｘ軸方向の微小幅*/
double k = 0.01; /*ｔ軸方向の微小幅*/

double simpson(double temp[N]){
  double s; /*Simpson積分でノルムの二乗を求めるために各配列を二乗する*/
  double sum1 = 0.0; /*4倍する方*/
  double sum2 = 0.0; /*2倍する方*/

  for (int i = 1; i <= nx-2;i=i+2){
    sum1 += temp[i]*temp[i];
    sum2 += temp[i + 1]*temp[i+1];
  }
  s = (h/3.0)*(temp[0]*temp[0] + 4 * sum1 + 2 * sum2 + temp[nx-1]*temp[nx-1]);
  return s;
}
double kikaku(double x, double norm){return x/(sqrt(norm)) ; }/*規格化(ノルムで割る)*/

double make_a(double x,double mu,double V,double gN){return -2+h*h*(mu-V-gN*pow(fabs(x),2)-(2/k)) ; }

double make_b(double x2, double x1, double x0, double mu,double V,double gN){ return -x2 - (-2 + h*h*(mu - V - gN*pow(fabs(x1), 2) + (2 / k)))*x1-x0; }

/*BdGで使うもの*/

double alpha =1 / (h*h);
double beta(double V, double mu, double x,double gN){ return 2 * (1 / (h*h)) + V - mu + 2 * gN*pow(fabs(x), 2); }
double ganma(double x,double gN){ return gN*x*x; }

double inner(double temp1[2*N],double temp2[2*N]){ /*不定形量内積*/
  double s;
  double sum1 = 0.0; /*4倍する方*/
  double sum2 = 0.0; /*2倍する方*/
	
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


double serchmax(double temp[2*N]){ /*配列の中から最大値を見つける*/
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
	

  /*BdGで使用*/
  //double T[2*N][2*N] = {};
  double Inn;

  stringstream FileName;
  FileName << "BdG.txt";
  ofstream ofs(FileName.str().c_str());
	
  for (int num = 360; num < 380;num++){
    gN = 0.01*num;

    double* psi0 = new double[N];/*初期配列ψ(x,0),その後はGaussSeidel法の初期値解として使用*/
    double* psi_t = new double[N];/*規格化した配列ψ(x,t)*/
    double* b = new double[N]();


    /*GPで定常解を出す*/
    for (int i = 0; i < nx; i++){ /*初期配列ψ(x,0)の作成*/
      psi0[i] = int_psi((-L / 2) + i*h);
    }

    norm = simpson(psi0);

    for (int i = 0; i < nx; i++){ /*配列ψの規格化*/
      psi_t[i] = kikaku(psi0[i], norm);
      psi0[i] = 0; /*ψ(x,t+1)の初期配列*/
    }

    mu = 3;/*μの初期値を設定*/

    for (int t = 0; k*t < Time; t++){ /*時間発展*/

      /*stringstream FileName;
	FileName << "GP" << t << ".txt";
	ofstream ofs(FileName.str().c_str());
	for (int i = 0; i < nx; i++){
	ofs << (-L / 2) + i*h << "\t" << psi_t[i] * psi_t[i] << endl;
	}*/




      /*行列a（式2.25）の左のやつと、ベクトルｂ（2.25の右のやつ）*/
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



      /*GaussSeidel法で時間が一つ進んだψを求める。psi0に格納(境界条件によりpsi0[0]=psi0[nx-1]=0)*/
      psi0[0] = 0.0;
      psi0[nx - 1] = 0.0;

      int j = 0;/*カウンタ*/

      while (j < 500){

	j += 1;
	for (int i = 1; i <= nx - 2; i++){
	  psi0[i] = (1 / a[i][i])*(b[i] - a[i][i - 1] * psi0[i - 1] - a[i][i + 1] * psi0[i + 1]);
	}
      }/*ψ(x,t+1)のGaussSeidel法で収束した解が得られてるはず*/
      /**/

      /*ψの規格化*/
      norm = simpson(psi0);
      for (int i = 0; i < nx; i++){ /*配列ψの規格化*/
	psi_t[i] = kikaku(psi0[i], norm);
	psi0[i] = 0;
      }

      /*μを補正*/
      mu = mu - (1 / (2 * k))*(norm - 1);
      //printf("mu=%f\n",mu);
    }
    printf("mu=%f\n",mu);

    /*
      stringstream FileName;
      FileName << "GP.txt";
      ofstream ofs(FileName.str().c_str());
      for (int i = 0; i < nx; i++){ //各ｘごとのψ二乗を出力
      ofs << (-L / 2) + i*h << "\t" << psi_t[i] * psi_t[i] << endl;
      }*/

    delete[] psi0;
    delete[] b;

    /************/
    /*続いてBdG*/
    /**********/




    /*行列Tの作成*/

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
    /*lapackに渡すべく配列に直す・・・*/
    /*左上*/


    T[0] = beta(pot(-L / 2), mu, psi_t[0], gN);
    T[2 * N] = -1 * alpha;
    T[2 * N*N - 3 * N - 1] = -1 * alpha;
    T[2 * N*N - N - 1] = beta(pot((-L / 2) + (N - 1)*h), mu, psi_t[N - 1], gN);


    /*左下*/


    T[N] = -1 * ganma(psi_t[0], gN);
    T[2 * N*N - 1] = -1 * ganma(psi_t[N - 1], gN);


    /*右上*/


    T[2 * N*N] = ganma(psi_t[0], gN);
    T[4 * N*N - N - 1] = ganma(psi_t[N - 1], gN);


    /*右下*/


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

    /*y0を作り、ｚを導出する(y0は解であるｚに上書きされるため最初から配列をｚとする。)*/


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

    /*先の計算でTは上書きされてる、Iも求まったので行列Sを構築する。Tの対角要素からIを引く？*/
    double* S = new double[4 * N*N]();

    /*左上*/
    S[0] = beta(pot(-L / 2), mu, psi_t[0], gN);
    S[2 * N] = -1 * alpha;
    S[2 * N*N - 3 * N - 1] = -1 * alpha;
    S[2 * N*N - N - 1] = beta(pot((-L / 2) + (N - 1)*h), mu, psi_t[N - 1], gN);


    /*左下*/
    S[N] = -1 * ganma(psi_t[0], gN);
    S[2 * N*N - 1] = -1 * ganma(psi_t[N - 1], gN);


    /*右上*/
    S[2 * N*N] = ganma(psi_t[0], gN);
    S[4 * N*N - N - 1] = ganma(psi_t[N - 1], gN);


    /*右下*/
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
