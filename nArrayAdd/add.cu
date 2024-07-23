#include "add.cuh"

namespace py = pybind11;

py::array_t<double> add_array_3d(py::array_t<double>& a, py::array_t<double>& b) {

	py::buffer_info buf1 = a.request();
	py::buffer_info buf2 = b.request();

	if (buf1.ndim != 3 || buf2.ndim != 3)
		throw std::runtime_error("numpy array dim must is 3!");

	for (int i = 0; i < buf1.ndim; i++)
	{
		if (buf1.shape[i]!=buf2.shape[i])
		{
		    throw std::runtime_error("inputs shape must match!");
		}
	}

	auto result = py::array_t<double>(buf1.size);
	result.resize({ buf1.shape[0], buf1.shape[1], buf1.shape[2] });
	py::buffer_info buf_result = result.request();

	double* ptr1 = (double*)buf1.ptr;
	double* ptr2 = (double*)buf2.ptr;
	double* ptr_result = (double*)buf_result.ptr;

	
	int N = buf1.size;
	size_t bytesCount = N * sizeof(double);

	int block_size = 256;
	int grid_size = (N + block_size - 1) / block_size;

	double* ipDevice_a, * ipDevice_b, * ipDevice_c;
	ErrorCheck(cudaMalloc((double**)&ipDevice_a, bytesCount));
	ErrorCheck(cudaMalloc((double**)&ipDevice_b, bytesCount));
	ErrorCheck(cudaMalloc((double**)&ipDevice_c, bytesCount));

	ErrorCheck(cudaMemcpy(ipDevice_a, ptr1, bytesCount, cudaMemcpyHostToDevice));
	ErrorCheck(cudaMemcpy(ipDevice_b, ptr2, bytesCount, cudaMemcpyHostToDevice));

	addArrayCuda<<<grid_size, block_size>>> (ipDevice_a, ipDevice_b, ipDevice_c, N);

	ErrorCheck(cudaMemcpy(ptr_result, ipDevice_c, bytesCount, cudaMemcpyDeviceToHost));

	ErrorCheck(cudaFree(ipDevice_a));
	ErrorCheck(cudaFree(ipDevice_b));
	ErrorCheck(cudaFree(ipDevice_c));

	return result;

}


__global__ void addArrayCuda(double* a, double* b, double* c, size_t N)
{
	int tid = threadIdx.x;
	int id = tid + blockIdx.x * blockDim.x;

	if (id < N)
	{
		c[id] = a[id] + b[id] + 0.1;
	}

}


PYBIND11_MODULE(nArrayAddCuda, m) {
	m.doc() = "Add two nArray";
	m.def("add_array_3d", &add_array_3d, "test func");
}

