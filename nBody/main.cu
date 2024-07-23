#include "main.cuh"

#ifndef SINGLE_PREC
#ifndef DOUBLE_PREC
#define SINGLE_PREC
#endif
#endif


#ifdef SINGLE_PREC

typedef float real;
#define MAX_DIST    200.0f
#define MAX_SPEED   100.0f
#define MASS        4.0f
#define DT          0.0001f
#define LIMIT_DIST  0.000001f
#define POW(x,y)    powf(x,y)
#define SQRT(x)     sqrtf(x)

#else

typedef double real;
#define MAX_DIST    200.0
#define MAX_SPEED   100.0
#define MASS        2.0
#define DT          0.00001
#define LIMIT_DIST  0.000001
#define POW(x,y)    pow(x,y)
#define SQRT(x)     sqrt(x)

#endif // SINGLE_PREC


typedef struct { real x, y, z; } Body;


void randomizedBodies(real* data, int n)
{
	for (int i = 0; i < n; ++i)
	{
		data[i] = 200.f * (rand() / (float)RAND_MAX) - 1.f;
	}
}



__global__ void nBody_updateVelocity(Body* position, Body* velocity, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	real ax = 0.f;
	real ay = 0.f;
	real az = 0.f;

	if (tid >= N) return;

	int cur_id = (tid + 1) % N;

	while (cur_id != tid)
	{
		real dx = position[cur_id].x - position[tid].x;
		real dy = position[cur_id].y - position[tid].y;
		real dz = position[cur_id].z - position[tid].z;

		real dist2 = dx * dx + dy * dy + dz * dz;
		if (dist2 < LIMIT_DIST) dist2 = LIMIT_DIST;

		real invdist = rsqrtf(dist2);
		real invdist3 = invdist * invdist * invdist;

		real s = MASS * invdist3;

		ax += dx * s;
		ay += dy * s;
		az += dz * s;

		cur_id = (cur_id + 1) % N;
	}

	velocity[tid].x = velocity[tid].x + ax;
	velocity[tid].y = velocity[tid].y + ay;
	velocity[tid].z = velocity[tid].z + az;
}


__global__ void nBody_updatePosition(Body* position, Body* Velocity, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= N) return;

	position[tid].x += (Velocity[tid].x * DT);
	position[tid].y += (Velocity[tid].y * DT);
	position[tid].z += (Velocity[tid].z * DT);

}



int launch()
{

#ifdef SINGLE_PREC
	printf("Using single-precision floating-point values\n");
#else // SINGLE_PREC
	printf("Using double-precision floating-point values\n");
#endif // SINGLE_PREC

	setGPU();

	// inital the data
	int i;
	int N = 1 << 14;
	int blockSize = 256;
	int gridSize = (N + blockSize - 1) / blockSize;
	int nBytes = 3 * N * sizeof(real);

	int iter, niters = 100;

	float* d_bodyPostionsBuf, * d_bodyVelocityBuf;
	float* h_bodyPostionsBuf;

	ErrorCheck(cudaMallocHost((void**)&h_bodyPostionsBuf, nBytes));
	randomizedBodies(h_bodyPostionsBuf, 3 * N);

	ErrorCheck(cudaMalloc((void **)&d_bodyPostionsBuf, nBytes));
	ErrorCheck(cudaMalloc((void**)&d_bodyVelocityBuf, nBytes));

	ErrorCheck(cudaMemset(d_bodyVelocityBuf, 0., nBytes));
	ErrorCheck(cudaMemcpy(d_bodyPostionsBuf, h_bodyPostionsBuf, nBytes, cudaMemcpyHostToDevice));

	Body* d_bodyPostions = (Body*)d_bodyPostionsBuf;
	Body* d_bodyVelocity = (Body*)d_bodyVelocityBuf;
	Body* h_bodyPostions = (Body*)h_bodyPostionsBuf;
	printf("\n");
	for (i = 0; i < 10; ++i)
	{
		printf("x y z: %8.3f  %8.3f  %8.3f\n", h_bodyPostions[i].x, h_bodyPostions[i].y,
			h_bodyPostions[i].z);
	}

	for (iter = 0; iter < niters; ++iter)
	{
		nBody_updateVelocity <<<gridSize, blockSize >>> (d_bodyPostions, d_bodyVelocity, N);
		nBody_updatePosition <<<gridSize, blockSize >>> (d_bodyPostions, d_bodyVelocity, N);
	}
	ErrorCheck(cudaMemcpy(h_bodyPostions, d_bodyPostions, nBytes, cudaMemcpyDeviceToHost));

	printf("\n");
	for (i = 0; i < 10; ++i)
	{
		printf("x y z: %8.3f  %8.3f  %8.3f\n", h_bodyPostions[i].x, h_bodyPostions[i].y,
			h_bodyPostions[i].z);
	}
	

	// free the data
	cudaFreeHost(h_bodyPostionsBuf);

	cudaFree(d_bodyPostionsBuf);
	cudaFree(d_bodyVelocityBuf);

	std::cout << "zzz" << std::endl;

	return 0;

}


PYBIND11_MODULE(nBody, m)
{
	m.doc() = "A cuda nbody test";
	m.def("launch", &launch, "A test function");
}
