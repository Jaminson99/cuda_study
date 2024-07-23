#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include "common.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

#define DLL_EXPORT_API __declspec(dllexport)
DLL_EXPORT_API int launch();

#ifdef __cplusplus
}
#endif
