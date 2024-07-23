#include "common.cuh"

#define BLOCKX 16
#define BLOCKY 16


// mode 0 is row-order, mode 1 is column-order
template <const int mode>
__global__ void transposeUnroll4(float* in, float* out, const int nx, const int ny)
{
	unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
	unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;

	unsigned int ti, to;
	if (mode == 0)
	{
		ti = iy * nx + ix;
		to = ix * ny + iy;

		if (ix + 3 * blockDim.x < nx && iy < ny)
		{
			out[to] = in[ti];
			out[to + ny * blockDim.x] = in[ti + blockDim.x];
			out[to + ny * blockDim.x * 2] = in[ti + blockDim.x * 2];
			out[to + ny * blockDim.x * 3] = in[ti + blockDim.x * 3];
		}
	}
	else
	{
		to = iy * nx + ix;
		ti = ix * ny + iy;

		if (ix + 3 * blockDim.x < nx && iy < ny)
		{
			out[to] = in[ti];
			out[to + blockDim.x] = in[ti + blockDim.x * ny];
			out[to + blockDim.x * 2] = in[ti + ny * blockDim.x * 2];
			out[to + blockDim.x * 3] = in[ti + ny * blockDim.x * 3];
		}
	}
}


// mode 0 is row-order, mode 1 is column-order
template <const int mode>
__global__ void copyUnroll4(float* in, float* out, const int nx, const int ny)
{
	unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
	unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;

	unsigned int ti, to;
	if (mode == 0)
	{
		ti = iy * nx + ix;

		if (ix + 3 * blockDim.x < nx && iy < ny)
		{
			out[ti] = in[ti];
			out[ti + blockDim.x] = in[ti + blockDim.x];
			out[ti + blockDim.x * 2] = in[ti + blockDim.x * 2];
			out[ti + blockDim.x * 3] = in[ti + blockDim.x * 3];
		}
	}
	else
	{
		ti = ix * ny + iy;

		if (ix + 3 * blockDim.x < nx && iy < ny)
		{
			out[ti] = in[ti];
			out[ti + blockDim.x * ny] = in[ti + blockDim.x * ny];
			out[ti + ny * blockDim.x * 2] = in[ti + ny * blockDim.x * 2];
			out[ti + ny * blockDim.x * 3] = in[ti + ny * blockDim.x * 3];
		}
	}
}


__global__ void copyGmem(float* in, float* out, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny) out[iy * nx + ix] = in[iy * nx + ix];
}


__global__ void transposeSmem(float* in, float* out, const int nx, const int ny)
{
	__shared__ float tile[BLOCKY][BLOCKX];

	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int in_idx = iy * nx + ix;

	unsigned int block_idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int transed_iy = block_idx / blockDim.y;  //index in transposed block
	unsigned int transed_ix = block_idx % blockDim.y;


	unsigned int ix2 = blockIdx.y * blockDim.y + transed_ix;  //index in transposed matrix
	unsigned int iy2 = blockIdx.x * blockDim.x + transed_iy;

	unsigned int out_idx = iy2 * ny + ix2;

	if (ix < nx && iy < ny)
	{
		tile[threadIdx.y][threadIdx.x] = in[in_idx];

		__syncthreads();

		out[out_idx] = tile[transed_ix][transed_iy];
	}
}


__global__ void transposeSmemUnroll2(float* in, float* out, const int nx, const int ny)
{
	__shared__ float tile[BLOCKY * BLOCKX * 2];

	unsigned int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int in_idx = iy * nx + ix;

	unsigned int block_idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int transed_iy = block_idx / blockDim.y;  //index in transposed block
	unsigned int transed_ix = block_idx % blockDim.y;


	unsigned int ix2 = blockIdx.y * blockDim.y + transed_ix;  //index in transposed matrix
	unsigned int iy2 = 2 * blockIdx.x * blockDim.x + transed_iy;

	unsigned int out_idx = iy2 * ny + ix2;

	if (ix+blockDim.x < nx && iy < ny)
	{
		unsigned int row_idx = threadIdx.y * (blockDim.x * 2) + threadIdx.x;
		tile[row_idx] = in[in_idx];
		tile[row_idx + BLOCKX] = in[in_idx + BLOCKX];

		__syncthreads();

		unsigned int col_idx = transed_ix * (blockDim.x * 2) + transed_iy;
		out[out_idx] = tile[col_idx];
		out[out_idx + ny * BLOCKX] = tile[col_idx + BLOCKX];
	}
}



int main()
{
	setGPU();

	int nx = 1 << 12;
	int ny = 1 << 12;
	int nBytes = nx * ny * sizeof(float);

	int blockx = BLOCKX;
	int blocky = BLOCKY;

	int unroll = 2;

	dim3 block(blockx, blocky);
	dim3 grid((nx + blockx* unroll - 1) / (blockx* unroll), (ny + blocky - 1) / blocky);

	printf("grid shape is [%d, %d], block shape is [%d, %d]\n", grid.x, grid.y, block.x, block.y);

	float* host_A, * host_B;
	host_A = (float*)malloc(nBytes);
	host_B = (float*)malloc(nBytes);
	
	for (int i = 0; i < nx; i++) for (int j = 0; j < ny; j++) host_A[i + j * ny] = j * 10.F + i;

	int repeat_num = 40;
	//transpose on cpu
	double begin_time = get_time();
	for (int n = 0; n < repeat_num; n++)
	{
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				host_B[i * ny + j] = host_A[j * nx + i];
			}
		}
	}
	double end_time = get_time();
	printf("time on cpu is %fs\n", end_time - begin_time);
	double cpu_cost = end_time - begin_time;

	float* device_A, * device_B;
	cudaMalloc((float**)&device_A, nBytes);
	cudaMalloc((float**)&device_B, nBytes);

	cudaMemcpy(device_A, host_A, nBytes, cudaMemcpyHostToDevice);

	//warm up
	transposeSmem <<<grid, block>>> (device_A, device_B, nx, ny);
	cudaDeviceSynchronize();

	begin_time = get_time();
	for (int n = 0; n < repeat_num; n++)
	{
		//transposeUnroll4<0> <<<grid, block>>> (device_A, device_B, nx, ny);
		//transposeSmem <<<grid, block>>> (device_A, device_B, nx, ny);
		transposeSmemUnroll2 <<<grid, block>>> (device_A, device_B, nx, ny);
		//copyGmem <<<grid, block>>> (device_A, device_B, nx, ny);
	}
	cudaDeviceSynchronize();
	end_time = get_time();
	printf("time on cpu is %fs\n", end_time - begin_time);
	double gpu_cost = end_time - begin_time;
	printf("accelerate %f times\n", cpu_cost / gpu_cost);

	double mem_bnd = 2.F * nx * ny * sizeof(float) / 1e9 / (gpu_cost / repeat_num);
	printf("the effective bandwidth is %f GB/s\n", mem_bnd);

	cudaMemcpy(host_B, device_B, nBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++) printf("%.1f, ", host_A[i]);
	printf("\n");
	for (int i = 0; i < 10; i++) printf("%.1f, ", host_B[i]);
	printf("\n");
	for (int i = 0; i < 10; i++) printf("%.1f, ", host_B[4095-i]);

	cudaFree(device_A);
	cudaFree(device_B);
	free(host_A);
	free(host_B);

	cudaDeviceReset();

	return -1;

}
