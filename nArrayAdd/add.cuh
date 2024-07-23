#ifndef _ADD_CUH_
#define _ADD_CUH_


#include "common.cuh"
#include "../pybind11/pybind11.h"
#include "../pybind11/numpy.h"

namespace py = pybind11;

__global__ void addArrayCuda(double* a, double* b, double* c, size_t N);

//#ifdef __cplusplus
//extern "C"
//{
//#endif

py::array_t<double> add_array_3d(py::array_t<double>& a, py::array_t<double>& b);

//#ifdef __cplusplus
//}
//#endif


#endif // !_ADD_CUH_
