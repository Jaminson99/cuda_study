#include "add.cuh"


py::array_t<double> add_array_3d(py::array_t<double>& a, py::array_t<double>& b) {

	py::buffer_info buf1 = a.request();
	py::buffer_info buf2 = b.request();

	if (buf1.ndim != 3 || buf2.ndim != 3)
		throw std::runtime_error("numpy array dim must is 3!");

	for (int i = 0; i < buf1.ndim; i++)
	{
		if (buf1.shape[i] != buf2.shape[i])
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

	deviceVector vDevice_a(N), vDevice_b(N), vDevice_c(N);
	double* prtDevice_a = thrust::raw_pointer_cast(&vDevice_a[0]);
	double* prtDevice_b = thrust::raw_pointer_cast(&vDevice_b[0]);
	double* prtDevice_c = thrust::raw_pointer_cast(&vDevice_c[0]);


	hostVector vHost_a = hostVector(ptr1, ptr1 + N);
	hostVector vHost_b = hostVector(ptr2, ptr2 + N);
	hostVector vHost_c = hostVector(ptr_result, ptr_result + N);

	vDevice_a = vHost_a;
	vDevice_b = vHost_b;

	addArrayCuda <<<grid_size, block_size>>> (prtDevice_a, prtDevice_b, prtDevice_c, N);

	vHost_a = vDevice_c;
	vDevice_a = vHost_a;

	addArrayCuda <<<grid_size, block_size>>> (prtDevice_a, prtDevice_b, prtDevice_c, N);

	vHost_c = vDevice_c;
	memcpy(ptr_result, &vHost_c[0], bytesCount);

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



PYBIND11_MODULE(nArrayAddCudaThrust, m) {
	m.doc() = "Add two nArray";
	m.def("add_array_3d", &add_array_3d, "test func");
}
