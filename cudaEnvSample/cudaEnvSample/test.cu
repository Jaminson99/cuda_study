#include "common.cuh"


__global__ void addGPU(int* A, int* B, int* C, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void reductionVector(int* g_idata, int* g_odata, const int nx)
{   
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + tid;
    if (id < nx)
    {
        sdata[tid] = g_idata[id];
    }
    else
    {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x) sdata[index] += sdata[index + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reductionVectorNoCfl(int* g_idata, int* g_odata, const int nx)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x * 2 + tid;
    if (id < nx)
    {
        sdata[tid] = g_idata[id] + g_idata[id + blockDim.x];
    }
    else
    {
        sdata[tid] = 0;
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
        g_odata[blockIdx.x] = sdata[0] + sdata[1];
    }
}


int main(void)
{
    setGPU();

    int nx = 65000*200;
    int ny = 1;
    unsigned int nxy = nx * ny;
    size_t stBytesCount = nxy * sizeof(int);

    int* ipHost_A, * ipHost_C, * ipHost_B;
    ipHost_A = (int*)malloc(stBytesCount);
    ipHost_B = (int*)malloc((stBytesCount + 255) / 256);
    ipHost_C = (int*)malloc(256 * sizeof(int));

    if (ipHost_A != NULL && ipHost_C != NULL && ipHost_B != NULL)
    {
        for (int i = 0; i < nxy; i++)
        {
            ipHost_A[i] = 1;
        }
        for (int i = 0; i < nxy/256; i++)
        {
            ipHost_B[i] = 0;
        }
        memset(ipHost_C, 0, 256 * sizeof(int));
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    int* ipDevice_A, * ipDevice_C, * ipDevice_B;
    ErrorCheck(cudaMalloc((int**)&ipDevice_A, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&ipDevice_B, (stBytesCount+255)/256), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&ipDevice_C, 256 * sizeof(int)), __FILE__, __LINE__);

    if (ipDevice_A != NULL && ipDevice_C != NULL && ipDevice_B != NULL)
    {
        ErrorCheck(cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount/256, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(ipDevice_C, ipHost_C, 256 * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    }
    else
    {
        printf("Fail to allocate memory\n");
        free(ipHost_A);
        free(ipHost_C);
        exit(1);
    }

    dim3 block(256, 1);
    dim3 grid((nx + block.x * 2 - 1) / block.x / 2, (ny + block.y - 1) / block.y);
    int nx2 = (nx + block.x*2 - 1) / 256 / 2;
    dim3 grid2((nx2 + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    printf("Thread config:grid:<%d, %d>, block:<%d, %d>\n", grid.x, grid.y, block.x, block.y);

    
    float begin_time_cpu = get_time();
    int res = 0;
    for (int j = 0; j < 500; j++)
    {   
        res = 0;
        for (long i = 0; i < nxy; i++)
        {   
            res += ipHost_A[i];
        }
    }
    float end_time_cpu = get_time();
    float cpu_cost = end_time_cpu - begin_time_cpu;
    printf("sum result on cpu is %d, time cost is %f\n", res, cpu_cost);

    cudaDeviceSynchronize();
    float begin_time = get_time();
    for (int i = 0; i < 500; i++)
    {
        //reductionVector <<<grid, block, 256>>> (ipDevice_A, ipDevice_B, nx);
        //reductionVector <<<grid2, block, 256>>> (ipDevice_B, ipDevice_C, nx2);
        //reductionVector <<<1, 256, 256>>> (ipDevice_C, ipDevice_C, 256);
        reductionVectorNoCfl<<<grid, block, 256>>> (ipDevice_A, ipDevice_B, nx);
        reductionVectorNoCfl<<<grid2, block, 256>>> (ipDevice_B, ipDevice_C, nx2);
        reductionVectorNoCfl<<<1, 256, 256>>> (ipDevice_C, ipDevice_C, nx);
    }
    cudaDeviceSynchronize();
    float end_time = get_time();
    float gpu_cost = end_time - begin_time;
    ErrorCheck(cudaMemcpy(ipHost_C, ipDevice_C, 256 * sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    
    printf("sum result is %d, time cost is %f\n", ipHost_C[0], gpu_cost);
    printf("accelerate %f time", cpu_cost / gpu_cost);

    free(ipHost_A);
    free(ipHost_C);

    ErrorCheck(cudaFree(ipDevice_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(ipDevice_C), __FILE__, __LINE__);

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
    return 0;
}   
