#include "common.cuh"


__global__ void conv(float* img, float* kernel, float* result, int width, int height, int kernelSize)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	int id = threadId + blockId * blockDim.x;

	int row = id / width;
	int col = id % width;

	if (id < width * height)
	{	
		float val = 0;
		for (int i = 0; i < kernelSize; ++i)
		{
			for (int j = 0; j < kernelSize; ++j)
			{
				int curRow = row - kernelSize / 2 + i;
				int curCol = col - kernelSize / 2 + j;

				int kernelId = j + i * kernelSize;

				if (curRow < 0 || curCol < 0 || curRow >= height || curCol >= width) {}
				else
				{
					val += img[curRow * width + curCol] * kernel[kernelId];
				}
			}
		}
		result[id] = val;
	}

}


int main(void)
{
	setGPU();

	// initial on CPU
	int width = 1024;
	int height = 1024;
	float* imgHost = new float[width * height];
	float* resultHost = new float[width * height];
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			imgHost[col + row * width] = (col + row) % 256;
		}
	}

	int kernelSize = 3;
	float* kernelHost = new float[kernelSize * kernelSize];
	for (int i = 0; i < kernelSize * kernelSize; ++i)
	{
		kernelHost[i] = i % kernelSize - 1.F;
	}

	printf("\n");
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			printf("%3.0f", kernelHost[j + i * kernelSize]);
		}
		printf("\n");
	}


	// initial on GPU
	float* imgDevice, * kernelDevice, * resultDevice;
	cudaMalloc((void**)&imgDevice, width * height * sizeof(float));
	cudaMalloc((void**)&kernelDevice, kernelSize * kernelSize * sizeof(float));
	cudaMalloc((void**)&resultDevice, width * height * sizeof(float));

	cudaMemcpy(imgDevice, imgHost, width * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernelDevice, kernelHost, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);


	// launch kernel
	int blockSize = 256;
	int gridSize = (width * height + blockSize - 1) / blockSize;
	conv <<<gridSize, blockSize>>> (imgDevice, kernelDevice, resultDevice, width, height, kernelSize);

	cudaMemcpy(resultHost, resultDevice, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	printf("\n");
	for (int i = 0; i < 10; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			printf("%3.0f", imgHost[j + i * width]);
		}
		printf("\n");
	}

	printf("\n");
	for (int i = 0; i < 10; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			printf("%3.0f", resultHost[j + i * width]);
		}
		printf("\n");
	}

	free(imgHost);
	free(kernelHost);
	free(resultHost);

	cudaFree(imgDevice);
	cudaFree(kernelDevice);
	cudaFree(resultDevice);

}
