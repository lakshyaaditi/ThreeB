Program 1:
#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
void merge(int* a, int l, int mid, int r)
{
int n1 = mid-l+1;
int n2 = r-mid;
int b[n1], c[n2];
int k = l;
for(inti = 0; i<n1; i++)
b[i] = a[k++];
for(inti = 0; i<n2; i++)
c[i] = a[k++];
k = l;
inti = 0, j = 0;
while(i<n1 & j<n2)
{
if(b[i]<c[j])
{
a[k++] = b[i++];
}
else
{
a[k++] = c[j++];
}
}
while(i<n1)
a[k++] = b[i++];
while(j<n2)
a[k++] = c[j++];
}
void mergesort(int* a, int l, int r)
{
if(l<r)
{
int mid;
#pragma omp parallel sections
{
mid = (l+r)/2;
#pragma omp section
{
//printf("thread id = %d\t l=%d\t mid=%d\n",omp_get_thread_num(),l,mid);
mergesort(a, l, mid);
}
#pragma omp section
{
//printf("thread id = %d\t r=%d\t mid+1=%d\n",omp_get_thread_num(),r,mid+1);
mergesort(a, mid+1, r);
}
}
merge(a,l,mid,r);
}
}
int main()
{
omp_set_nested(1);
int start=1;
/*a=(int*)malloc(100*sizeof(int));
for(inti = 0; i<100; i++)
a[i] = rand()%1000;
mergesort(a,0,99);*/
printf("\n\nInput Size\t1\t2\t4\t8\t");
for(inti = 0; i<4; i++)
{
int size = start*10;
start = size;
int a[size];
for(int j = 0; j<size; j++)
{
a[j] = rand()%100000;
}
printf("\n\n%d\t",size);
for(inti = 0; i<4; i++)
{
omp_set_num_threads(2*(i));
double t1 = omp_get_wtime();
mergesort(a,0,size-1);
double t2 = omp_get_wtime();
printf("%lf\t",t2-t1);
}
}
return 0;
}

Program 2:
#include <stdio.h>
#include <omp.h>

/* Main Program */
main()
{
intNoofRows, NoofCols, Vectorsize, i, j;
/*float **Matrix, *Vector, *Result, *Checkoutput;*/
double **Matrix, *Vector, *Result, *Checkoutput;

printf("Read the matrix size noofrows and columns and vectorsize\n");
scanf("%d%d%d", &NoofRows, &NoofCols, &Vectorsize);

if (NoofRows<= 0 || NoofCols<= 0 || Vectorsize<= 0)
{
printf("The Matrix and Vectorsize should be of positive sign\n");
exit(1);
}

/* Checking For Matrix Vector Computation Necessary Condition */
if (NoofCols != Vectorsize)
{
printf("Matrix Vector computation cannot be possible \n");
exit(1);
}

/* Dynamic Memory Allocation And Initialization Of Matrix Elements */
/* Matrix = (float **) malloc(sizeof(float) * NoofRows); */

Matrix = (double **) malloc(sizeof(double) * NoofRows);
for (i = 0; i<NoofRows; i++)
 {
/* Matrix[i] = (float *) malloc(sizeof(float) * NoofCols); */
Matrix[i] = (double *) malloc(sizeof(double) * NoofCols);
for (j = 0; j <NoofCols; j++)
Matrix[i][j] = i + j;
}

/* Printing The Matrix */
printf("The Matrix is \n");
for (i = 0; i<NoofRows; i++)
{
for (j = 0; j <NoofCols; j++)
printf("%lf \t", Matrix[i][j]);
printf("\n");
}
printf("\n");

/* Dynamic Memory Allocation */
/*Vector = (float *) malloc(sizeof(float) * Vectorsize);*/
Vector = (double *) malloc(sizeof(double) * Vectorsize);

/* vector Initialization */
for (i = 0; i<Vectorsize; i++)
Vector[i] = i;
printf("\n");

/* Printing The Vector Elements */
printf("The Vector is \n");
for (i = 0; i<Vectorsize; i++)
printf("%lf \t", Vector[i]);

/* Dynamic Memory Allocation */
/* Result = (float *) malloc(sizeof(float) * NoofRows);
Checkoutput = (float *) malloc(sizeof(float) * NoofRows); */
Result = (double *) malloc(sizeof(double) * NoofRows);
Checkoutput = (double *) malloc(sizeof(double) * NoofRows);
for (i = 0; i<NoofRows; i = i + 1)
{
Result[i]=0;
Checkoutput[i]=0;
}

/* OpenMP Parallel Directive */
omp_set_num_threads(32);
#pragma omp parallel for private(j)
for (i = 0; i<NoofRows; i = i + 1)
for (j = 0; j <NoofCols; j = j + 1)
Result[i] = Result[i] + Matrix[i][j] * Vector[j];

/* Serial Computation */
for (i = 0; i<NoofRows; i = i + 1)
for (j = 0; j <NoofCols; j = j + 1)
Checkoutput[i] = Checkoutput[i] + Matrix[i][j] * Vector[j];

/* Checking with the serial calculation */
for (i = 0; i<NoofRows; i = i + 1)
if (Checkoutput[i] == Result[i])
continue;
else
{
printf("There is a difference from Serial and Parallel Computation \n");
exit(1);
}
printf("\nThe Matrix Computation result is \n");
for (i = 0; i<NoofRows; i++)
printf("%lf \n", Result[i]);

/* Freeing The Memory Allocations */
free(Vector);
free(Result);
free(Matrix);
free(Checkoutput);
}

Output:
/PP_Lab$ gcc -fopenmp mul_matrixvector_lab10.c
/PP_Lab$ ./a.out mul_matrixvector_lab10
Read the matrix size noofrows and columns and vectorsize
3 3
3
Program 3:
#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int *prime_table ( intprime_num )
{
    //printf("prime table by thread %d", omp_get_thread_num());
  inti;
  int j;
  int p;
  int prime;
  int *primes;
  primes = ( int * ) malloc ( prime_num * sizeof ( int ) );
  i = 2;
  p = 0;
  while ( p <prime_num )
  {
    prime = 1;
    for ( j = 2; j <i; j++ )
    {
      if ( ( i % j ) == 0 )
      {
        prime = 0;
        break;
      }
    }
    if ( prime )
    {
      primes[p] = i;
      p = p + 1;
    }
    i = i + 1;
  }
  return primes;
}
double *sine_table ( intsine_num )
{
    //printf("sine table by thread %d", omp_get_thread_num());
  double a;
  inti;
  int j;
  double pi = 3.141592653589793;
  double *sines;
  sines = ( double * ) malloc ( sine_num * sizeof ( double ) );
  for ( i = 0; i<sine_num; i++ )
  {
    sines[i] = 0.0;
    for ( j = 0; j <= i; j++ )
    {
      a = ( double ) ( j ) * pi / ( double ) ( sine_num - 1 );
      sines[i] = sines[i] + sin ( a );
    }
  }
  return sines;
}
int main()
{
    omp_set_nested(1);
    int size=10;
    printf("\n\nInput Size\t1\t2\t4\t8\t");
    for(inti = 0; i<5; i++)
    {
        printf("\n\n%d\t",size);
        for(int x = 0; x<4; x++)
        {
            double t1 = omp_get_wtime();
            #pragma omp parallel sections
            {
                omp_set_num_threads(2*x);
                #pragma omp section
                {
                    int* a = (int*)malloc(size*sizeof(int));
                    a = prime_table(size);
                    /*for(int y=0; y<size; y++)
                    {
                        printf("%d\n",a[y]);
                    }*/
                }
                #pragma omp section
                {
                   double* b = (double*)malloc(size*sizeof(double));
                    b = sine_table(size);
                    for(int z=0; z<size; z++)
                    {
                        printf("%lf\n",b[z]);
                    }
                }
            }
            double t2 = omp_get_wtime();
            printf("%lf\t",t2-t1);
        }
        size = size*10;
    }
    return 0;
}
Program 4:
#include <stdio.h>
#include <malloc.h>
#include <omp.h>

long long factorial(long n)
{
long long i,out;
out = 1;
for (i=1; i<n+1; i++) out *= i;
return(out);
}

int main(int argc, char **argv)
{
int i,j,threads;
long long *x;
long long n=12;

/* Set number of threads equal to argv[1] if present */
if (argc > 1)
{
   threads = atoi(argv[1]);
   if (omp_get_dynamic())
   {
     omp_set_dynamic(0);
     printf("called omp_set_dynamic(0)\n");
    }
   omp_set_num_threads(threads);
 }

printf("%d threads\n",omp_get_max_threads());
x = (long long *) malloc(n * sizeof(long));
for (i=0;i<n;i++) x[i]=factorial(i);
j=0;




/* Is the output the same if the following line is commented out? */
#pragma omp parallel for firstprivate(x,j)
for (i=1; i<n; i++)
{
j += i;
x[i] = j*x[i-1];
}
for (i=0; i<n; i++)
printf("factorial(%2d)=%14lld x[%2d]=%14lld\n",i,factorial(i),i,x[i]);
return 0;
}

Output:

/PP_Lab$ gcc -fopenmp factorial_lab13.c
/PP_Lab$ ./a.out factorial_lab13
1 threads
factorial( 0)=             1 x[ 0]=             1
factorial( 1)=             1 x[ 1]=             1
factorial( 2)=             2 x[ 2]=             3
factorial( 3)=             6 x[ 3]=            18
factorial( 4)=            24 x[ 4]=           180
factorial( 5)=           120 x[ 5]=          2700
factorial( 6)=           720 x[ 6]=         56700
factorial( 7)=          5040 x[ 7]=       1587600
factorial( 8)=         40320 x[ 8]=      57153600
factorial( 9)=        362880 x[ 9]=    2571912000
factorial(10)=       3628800 x[10]=  141455160000
factorial(11)=      39916800 x[11]= 9336040560000


Program 5:
#include <iostream>
#include <cstdlib> // or <stdlib.h> rand, srand
#include <ctime> // or <time.h> time
#include <omp.h>
#include <math.h>
#define K 4
using namespace std;
intnum_threads;
longnum_points;
long** points; // 2D array points[x][0] -> point location, points[x][1] -> distance from cluster mean
int cluster[K][2] = {
    {75, 25}, {25, 25}, {25, 75}, {75, 75}
};
longcluster_count[K];
voidpopulate_points() {
    // Dynamically allocate points[num_points][2] 2D array
points = new long*[num_points];
for (long i=0; i<num_points; i++)
points[i] = new long[2];
    // Fill random points (0 to 100)
srand(time(NULL));
for (long i=0; i<num_points; i++) {
points[i][0] = rand() % 100;
points[i][1] = rand() % 100;
    }
    // Initialize cluster_count
for (inti=0; i<K; i++) {
cluster_count[i] = 0;
    }
}
doubleget_distance(int x1, int y1, int x2, int y2) {
int dx = x2-x1, dy = y2-y1;
return (double)sqrt(dx*dx + dy*dy);
}
voidclassify_points() {
    #pragma omp parallel for num_threads(num_threads)
for (long i=0; i<num_points; i++) {
doublemin_dist = 1000, cur_dist = 1;
intcluster_index = -1;
for (int j=0; j<K; j++) {
cur_dist = get_distance(
points[i][0], points[i][1],
cluster[j][0], cluster[j][1]
                       );
if (cur_dist<min_dist) {
min_dist = cur_dist;
cluster_index = j;
            }
        }
cluster_count[cluster_index]++;
    }
}
int main(intargc, char* argv[]) {
num_points = atol(argv[1]);
num_threads = atoi(argv[2]);
populate_points();
double t1 = omp_get_wtime();
classify_points();
double t2 = omp_get_wtime();
double t = (t2 - t1) * 1000;
cout<< "Time Taken: " << t << "ms" <<endl;
}
