#include "common.cuh"


__global__ void reductionInnerDot(float* vectorA, float* vectorB, float* vectorC, const int nx)
{
	extern __shared__  float sdata[];
	int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + tid;

	if (id < nx)
	{
		sdata[tid] = vectorA[id] * vectorB[id] * 0.9F;
	}
	else
	{
		sdata[tid] = 0.F;
	}
	__syncthreads();

	if (tid < 128) sdata[tid] += sdata[tid + 128];
	__syncthreads();

	if (tid < 64) sdata[tid] += sdata[tid + 64];
	__syncthreads();

	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		vectorC[blockIdx.x] = sdata[0] + sdata[1];
	}
}

template <int blockx>
	__global__ void reductionSum(float* inVector, float* outVector, const int nx)
	{
		extern __shared__ float sdata[];
		int tid = threadIdx.x;
		int id = blockDim.x * blockIdx.x + tid;

		if (id < nx)
		{
			sdata[tid] = inVector[id];
		}
		else
		{
			sdata[tid] = 0.F;
		}
		__syncthreads();

		if (blockx >= 256)
		{
			if (tid < 128) sdata[tid] += sdata[tid + 128];
			__syncthreads();
		}

		if (blockx >= 128)
		{
			if (tid < 64) sdata[tid] += sdata[tid + 64];
			__syncthreads();
		}

		if (tid < 32)
		{
			sdata[tid] += sdata[tid + 32];
			sdata[tid] += sdata[tid + 16];
			sdata[tid] += sdata[tid + 8];
			sdata[tid] += sdata[tid + 4];
			sdata[tid] += sdata[tid + 2];
			outVector[blockIdx.x] = sdata[0] + sdata[1];
		}
	}


int main(void)
{
	setGPU();

	int nx = 65000 * 100;
	int ny = 1;
	const int blockx = 256;
	unsigned int nxy = nx * ny;
	size_t stBytesCount = nxy * sizeof(float);

	dim3 block(blockx, 1);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	printf("Thread config:grid:<%d, %d>, block:<%d, %d>\n", grid.x, grid.y, block.x, block.y);

	int nx2 = (nx + block.x - 1) / block.x;
	dim3 grid2((nx2 + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	printf("Thread config:grid:<%d, %d>, block:<%d, %d>\n", grid2.x, grid2.y, block.x, block.y);
	
	float* ipHost_A, * ipHost_C, * ipHost_B, * ipHost_D;
	//ipHost_A = (float*)malloc(stBytesCount);
	//ipHost_B = (float*)malloc(stBytesCount);
	//ipHost_C = (float*)malloc((stBytesCount + blockx - 1) / blockx);
	//ipHost_D = (float*)malloc(blockx * sizeof(float));

	cudaHostAlloc((void**)&ipHost_A, stBytesCount, cudaHostAllocDefault);
	cudaHostAlloc((void**)&ipHost_B, stBytesCount, cudaHostAllocDefault);
	cudaHostAlloc((void**)&ipHost_C, (stBytesCount + blockx - 1) / blockx, cudaHostAllocDefault);
	cudaHostAlloc((void**)&ipHost_D, blockx * sizeof(float), cudaHostAllocDefault);

	printf("-----created ipHosts-----\n");

	if (ipHost_A != NULL && ipHost_C != NULL && ipHost_B != NULL && ipHost_D != NULL)
	{
		for (int i = 0; i < nxy; i++)
		{
			ipHost_A[i] = 0.5F;
			ipHost_B[i] = 0.5F;
		}
		memset(ipHost_C, 0, ((stBytesCount + blockx - 1) / blockx));
		memset(ipHost_D, 0, blockx * sizeof(float));
	}
	else
	{
		printf("Fail to allocate host memory!\n");
		exit(-1);
	}

	printf("-----allocated ipHosts-----\n");

	float* ipDevice_A, * ipDevice_C, * ipDevice_B, * ipDevice_D;
	ErrorCheck(cudaMalloc((float**)&ipDevice_A, stBytesCount), __FILE__, __LINE__);
	ErrorCheck(cudaMalloc((float**)&ipDevice_B, stBytesCount), __FILE__, __LINE__);
	ErrorCheck(cudaMalloc((float**)&ipDevice_C, (stBytesCount + blockx - 1) / blockx), __FILE__, __LINE__);
	ErrorCheck(cudaMalloc((float**)&ipDevice_D, blockx * sizeof(float)), __FILE__, __LINE__);

	printf("-----created ipDevices-----\n");

	if (ipDevice_A != NULL && ipDevice_C != NULL && ipDevice_B != NULL && ipDevice_D != NULL)
	{
		ErrorCheck(cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
		ErrorCheck(cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
		ErrorCheck(cudaMemcpy(ipDevice_C, ipHost_C, ((stBytesCount + blockx - 1) / blockx), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		ErrorCheck(cudaMemcpy(ipDevice_D, ipHost_D, blockx * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	}
	else
	{
		printf("Fail to allocate memory\n");
		free(ipHost_A);
		free(ipHost_B);
		free(ipHost_C);
		free(ipHost_D);
		exit(1);
	}
	printf("-----allocated ipDevices-----\n");

	float begin_time_cpu = get_time();
	double res = 0;
	for (int j = 0; j < 30; j++)
	{
		res = 0.F;
		for (long i = 0; i < nxy; i++)
		{
			res += (ipHost_A[i] * ipHost_B[i] * 0.9);
		}
	}
	float end_time_cpu = get_time();
	float cpu_cost = end_time_cpu - begin_time_cpu;
	printf("sum result on cpu is %f, time cost is %f\n", res, cpu_cost);


	cudaDeviceSynchronize();
	float begin_time = get_time();
	for (int i = 0; i < 30; i++)
	{
		reductionInnerDot <<<grid, block, blockx >>> (ipDevice_A, ipDevice_B, ipDevice_C, nx);
		reductionSum<blockx> <<<grid2, block, blockx >>> (ipDevice_C, ipDevice_D, nx2);
		reductionSum<128> <<<1, block, 128 >>> (ipDevice_D, ipDevice_D, 128);
	}
	cudaDeviceSynchronize();
	float end_time = get_time();
	float gpu_cost = end_time - begin_time;
	ErrorCheck(cudaMemcpy(ipHost_D, ipDevice_D, blockx * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	printf("sum result is %f, time cost is %f\n", ipHost_D[0], gpu_cost);
	printf("accelerate %f time", cpu_cost / gpu_cost);

	//free(ipHost_A);
	//free(ipHost_B);
	//free(ipHost_C);
	//free(ipHost_D);

	cudaFreeHost(ipHost_A);
	cudaFreeHost(ipHost_B);
	cudaFreeHost(ipHost_C);
	cudaFreeHost(ipHost_D);

	ErrorCheck(cudaFree(ipDevice_A), __FILE__, __LINE__);
	ErrorCheck(cudaFree(ipDevice_B), __FILE__, __LINE__);
	ErrorCheck(cudaFree(ipDevice_C), __FILE__, __LINE__);
	ErrorCheck(cudaFree(ipDevice_D), __FILE__, __LINE__);

	ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
	return 0;

}
