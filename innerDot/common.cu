#include "common.cuh"


double get_time(void)
{
	LARGE_INTEGER timer;
	static LARGE_INTEGER fre;
	static int init = 0;
	double t;

	if (init != 1) {
		QueryPerformanceFrequency(&fre);
		init = 1;
	}

	QueryPerformanceCounter(&timer);

	t = timer.QuadPart * 1. / fre.QuadPart;

	return t;
}


cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
	if (error_code != cudaSuccess)
	{
		printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line=%d\r\n",
			error_code, cudaGetErrorName(error_code),
			cudaGetErrorString(error_code), filename, lineNumber
		);
		exit(-1);
	}
	return error_code;
}


void setGPU()
{
	int deviceNum = 0;
	cudaError_t error = ErrorCheck(cudaGetDeviceCount(&deviceNum), __FILE__, __LINE__);

	if (error != cudaSuccess || deviceNum == 0)
	{
		printf("None CUDA compatible GPU found\n");
		exit(-1);
	}
	else
	{
		printf("The num of GPU is %d.\n", deviceNum);
	}

	int device = 0;
	error = ErrorCheck(cudaSetDevice(device), __FILE__, __LINE__);
	if (error != cudaSuccess)
	{
		printf("Fail to set GPU 0 for computing\n");
		exit(-1);
	}
	else
	{
		printf("Set GPU 0\n");
	}
}
