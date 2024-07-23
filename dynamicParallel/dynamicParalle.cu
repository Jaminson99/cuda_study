#include "common.cuh"

__constant__ int ipDevice_Const[1024];

__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
	int tid = threadIdx.x;
	int blockId = blockIdx.x;
	printf("Hello word from recursion %d, block %d, thread %d\n", iDepth, blockId, tid);

	if (iSize == 1) return;

	int nThreads = iSize >> 1;

	if (tid == 0 && nThreads > 0 && blockIdx.x == 0)
	{
		nestedHelloWorld <<<2, nThreads>>> (nThreads, ++iDepth);
		printf("-----------> nested execution depth: %d, nThreads: %d\n", iDepth, nThreads);
	}

}


__global__ void gpuRecursiveReduce2(int* g_idata, int* g_odata, int stride, int const dim)
{
	int *idata = g_idata + blockIdx.x * dim;

	if (stride == 1 && threadIdx.x == 0)
	{
		g_odata[blockIdx.x] = idata[0] + idata[1];
		return;
	}

	idata[threadIdx.x] += idata[threadIdx.x + stride];

	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		gpuRecursiveReduce2 <<<gridDim.x, stride/2>>> (g_idata, g_odata, stride/2, dim);
	}

}


int main(void)
{
	setGPU();

	int nx = 1024*256;
	int ny = 1;
	int nxy = nx * ny;

	size_t stBytesCount = nxy * sizeof(int);

	dim3 grid(2048, 1);
	int blockx = nx/grid.x;
	dim3 block(blockx, 1);

	int* ipHost_A, * ipHost_B;
	cudaHostAlloc((void**)&ipHost_A, stBytesCount, cudaHostAllocDefault);
	cudaHostAlloc((void**)&ipHost_B, grid.x * sizeof(int), cudaHostAllocDefault);

	for (int i = 0; i < nxy; i++) ipHost_A[i] = 1;

	int* ipDevice_A, * ipDevice_B;
	ErrorCheck(cudaMalloc((int**)&ipDevice_A, stBytesCount), __FILE__, __LINE__);
	ErrorCheck(cudaMalloc((int**)&ipDevice_B, grid.x * sizeof(int)), __FILE__, __LINE__);

	ErrorCheck(cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
	ErrorCheck(cudaMemcpy(ipDevice_B, ipHost_B, grid.x * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	int ipHost_a[1024];
	if (ipDevice_Const != NULL) ErrorCheck(cudaMemcpyToSymbol(ipDevice_Const, ipHost_a, 1024*sizeof(int), 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);

	double begin_timec_cpu = get_time();
	int res = 0;
	for (int i = 0; i < 100; i++)
	{
		res = 0;
		for (int j = 0; j < nxy; j++)
		{
			res += ipHost_A[j];
		}
	}
	double end_time_cpu = get_time();
	double cpu_cost = end_time_cpu - begin_timec_cpu;
	printf("sum result on cpu is %d, time cost is %f\n", res, cpu_cost);

	cudaDeviceSynchronize();
	double begin_time = get_time();
	res = 0;
	gpuRecursiveReduce2 << <grid, block >> > (ipDevice_A, ipDevice_B, block.x / 2, block.x);
	ErrorCheck(cudaMemcpy(ipHost_B, ipDevice_B, grid.x * sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	for (int j = 0; j < grid.x; j++) res += ipHost_B[j];
	for (int i = 0; i < 100-1; i++)
	{	
		res = 0;
		gpuRecursiveReduce2 <<<grid, block>>>(ipDevice_A, ipDevice_B, block.x/2, block.x);
		for (int j = 0; j < grid.x; j++) res += ipHost_B[j];
	}
	double end_time = get_time();
	double gpu_cost = end_time - begin_time;
	ErrorCheck(cudaMemcpy(ipHost_B, ipDevice_B, grid.x * sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	printf("sum result is %d, time cost is %f\n", res, gpu_cost);
	printf("accelerate %f time\n", cpu_cost / gpu_cost);

	cudaFreeHost(ipHost_A);
	cudaFreeHost(ipHost_B);

	ErrorCheck(cudaFree(ipDevice_A), __FILE__, __LINE__);
	ErrorCheck(cudaFree(ipDevice_B), __FILE__, __LINE__);

	ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
	return 0;

}
