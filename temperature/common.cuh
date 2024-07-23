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

cudaError_t Error_Check(cudaError_t error_code, const char* file, int line);
#define ErrorCheck( err ) (Error_Check( err, __FILE__, __LINE__ ))

void setGPU();
