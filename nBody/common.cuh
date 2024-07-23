#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Windows.h>
#include <cmath>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


double get_time();


#ifdef __cplusplus
extern "C"
{
#endif

	cudaError_t ErrorCheck(cudaError_t error_code);

#ifdef __cplusplus
}
#endif


void setGPU();
