#include "pybind11//pybind11.h"
#include "main_2DtexObj.cuh"

#pragma comment(lib, "temperature_makedll.dll")

//int main()
//{	
//	launch();
//	return 0;
//}


PYBIND11_MODULE(temp_cuda, m)
{
	m.doc() = "A cuda test";
	m.def("launch", &launch, "A test function");