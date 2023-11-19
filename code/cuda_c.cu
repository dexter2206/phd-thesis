#include <algorithm>
#include <iostream>
#include <valarray>

__global__
void addVec(float* x, float* y, float* res, int n) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   int gridsize = blockDim.x * gridDim.x;

   for(int i = tid; i < n; i += gridsize) { res[i] = x[i] + y[i]; }
}

int main() {
   const int N = 100000, nBlocks = 256, nThreads=128;
   std::valarray<float> x(N), y(N), res(N);
   float* x_d, * y_d, * res_d;
   int nBytes = N * sizeof(float);

   srand(1234);

   for(int i = 0; i < N; i++) {
      x[i] = float(rand()) / RAND_MAX;
      y[i] = float(rand()) / RAND_MAX;
   }

   cudaMalloc(&x_d, N * sizeof(float));
   cudaMalloc(&y_d, N * sizeof(float));
   cudaMalloc(&res_d, N * sizeof(float));

   cudaMemcpy(x_d, &x[0], nBytes, cudaMemcpyHostToDevice);
   cudaMemcpy(y_d, &y[0], nBytes, cudaMemcpyHostToDevice);

   addVec<<<nBlocks, nThreads>>>(x_d, y_d, res_d, N);

   cudaMemcpy(&res[0], res_d, nBytes, cudaMemcpyDeviceToHost);

   std::cout << "Max error: "
             << std::abs(res - (x + y)).max()
             << std::endl;
   cudaFree(x_d);
   cudaFree(y_d);
   cudaFree(res_d);
   return 0;
}
