#include "common.cuh"


__global__ void vectorSum(float* vectorA, float* vectorB, float* result, int length)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int id = tid + bid * blockDim.x;

	if (id < length)
	{
		result[id] = vectorA[id] + vectorB[id];
	}

}



int main(void)
{
	setGPU();
	int nLength = 1 << 15;
	int nBytes = nLength * sizeof(float);

	float* hostA, * hostB, * hostResult;
	cudaHostAlloc((void**)&hostA, nBytes, cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostB, nBytes, cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostResult, nBytes, cudaHostAllocDefault);
	for (int i = 0; i < nLength; ++i)
	{
		hostA[i] = nLength - i;
		hostB[i] = i;
	}

	float* deviceA, * deviceB, * deviceResult;
	cudaMalloc((void**)&deviceA, nBytes);
	cudaMalloc((void**)&deviceB, nBytes);
	cudaMalloc((void**)&deviceResult, nBytes);
	cudaMemcpy(deviceA, hostA, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, nBytes, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = nLength / blockSize;

	// create streams
	int n_stream = 4;
	cudaStream_t* streams = (cudaStream_t*)malloc(n_stream * sizeof(cudaStream_t));
	for (int i = 0; i < n_stream; ++i)
	{
		cudaStreamCreate(&streams[i]);
	}

	
	// warmup
	vectorSum <<<gridSize, blockSize>>> (deviceA, deviceB, deviceResult, nLength);

	// syn
	double beginTime = get_time();
	cudaMemcpy(deviceA, hostA, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, nBytes, cudaMemcpyHostToDevice);
	vectorSum <<<gridSize, blockSize>>> (deviceA, deviceB, deviceResult, nLength);
	cudaMemcpy(hostResult, deviceResult, nBytes, cudaMemcpyDeviceToHost);
	double endTime = get_time();
	printf("\noriginal time: %.8f", endTime - beginTime);


	// asyn
	int iLength = nLength / n_stream;
	int iBytes = nBytes / n_stream;
	beginTime = get_time();
	for (int i = 0; i < n_stream; ++i)
	{
		int ioffset = iLength * i;
		cudaMemcpyAsync(&deviceA[ioffset], &hostA[ioffset], iBytes,
			cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&deviceB[ioffset], &hostB[ioffset], iBytes,
			cudaMemcpyHostToDevice, streams[i]);
		vectorSum <<<gridSize/n_stream, blockSize, 0, streams[i]>>> (&deviceA[ioffset], &deviceB[ioffset], &deviceResult[ioffset], iLength);
		cudaMemcpyAsync(&hostResult[ioffset], &deviceResult[ioffset], iBytes,
			cudaMemcpyDeviceToHost, streams[i]);
	}
	endTime = get_time();
	printf("\nasyn time: %.8f", endTime - beginTime);

	printf("\n");
	for (int i = 0; i < 10; ++i) printf("%.1f ", hostResult[i]);

	cudaFreeHost(hostA);
	cudaFreeHost(hostB);
	cudaFreeHost(hostResult);

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceResult);
}

