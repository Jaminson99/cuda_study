#ifndef _ADD_CUH_
#define _ADD_CUH_


#include "common.cuh"
#include "../pybind11/pybind11.h"
#include "../pybind11/numpy.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef thrust::device_vector<double> deviceVector;
typedef thrust::host_vector<double> hostVector;

namespace py = pybind11;

__global__ void addArrayCuda(double* a, double* b, double* c, size_t N);
py::array_t<double> add_array_3d(py::array_t<double>& a, py::array_t<double>& b); 


#endif // !_ADD_CUH_

