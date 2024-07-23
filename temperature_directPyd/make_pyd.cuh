#ifndef MAKE_PYD_CUH
#define MAKE_PYD_CUH


#include "common.cuh"
#include "./common/book.h"
#include "./common/cpu_anim.h"
#include "./pybind11/pybind11.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define DLL_EXPORT_API __declspec(dllexport)
DLL_EXPORT_API int launch();


#ifdef __cplusplus
}
#endif

#endif // !MAKE_PYD_CUH

