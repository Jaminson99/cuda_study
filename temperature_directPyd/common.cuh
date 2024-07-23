#ifndef COMMON_CUH
#define COMMON_CUH


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Windows.h>
#include <cmath>
#include <math.h>
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


double get_time();

#ifdef __cplusplus
extern "C"
{
#endif

#define DLL_EXPORT_API __declspec(dllexport)
DLL_EXPORT_API cudaError_t Error_Check(cudaError_t error_code, const char* file, int line);


#ifdef __cplusplus
}
#endif

#define ErrorCheck( err ) (Error_Check( err, __FILE__, __LINE__ ))

void setGPU();

#endif // !COMMON_CUH
