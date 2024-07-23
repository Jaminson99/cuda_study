#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Windows.h>
#include<cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


double get_time(void);
cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber);
void setGPU();
