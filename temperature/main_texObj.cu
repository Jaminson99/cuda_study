#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include "common.cuh"
#include "./common/book.h"
#include "./common/cpu_anim.h"


#define DIM 1024
#define PI 3.1415926f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f



struct DataBlock {
	unsigned char* output_bitmap;
	float* dev_inSrc;
	float* dev_constSrc;
	float* dev_outSrc;
	CPUAnimBitmap* bitmap;

	cudaEvent_t start, stop;
	float totalTime;
	float frames;

	cudaTextureObject_t texIn;
	cudaTextureObject_t texOut;
	cudaTextureObject_t texConstSrc;
};


__global__ void blend_kernel(cudaTextureObject_t in, float* dst)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = x + y * (blockDim.x * gridDim.x);

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0)   left++;
	if (x == DIM - 1) right--;

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0)   top += DIM;
	if (y == DIM - 1) bottom -= DIM;

	float t, l, c, r, b;

	t = tex1Dfetch<float>(in, top);
	l = tex1Dfetch<float>(in, left);
	c = tex1Dfetch<float>(in, offset);
	r = tex1Dfetch<float>(in, right);
	b = tex1Dfetch<float>(in, bottom);

	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}


__global__ void copy_kernel(cudaTextureObject_t texConstSrc, float* iptr)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = x + y * (blockDim.x * gridDim.x);

	float c = tex1Dfetch<float>(texConstSrc, offset);
	if (c != 0) iptr[offset] = c;
}


void anim_gpu(DataBlock* d, int ticks)
{
	ErrorCheck(cudaEventRecord(d->start, 0));

	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	CPUAnimBitmap* bitmap = d->bitmap;

	volatile bool dstOut = true;
	for (int i = 0; i < 90; ++i)
	{
		float* in, * out;
		cudaTextureObject_t texIn;

		if (dstOut) {
			in = d->dev_inSrc;
			out = d->dev_outSrc;
			texIn = d->texIn;
		}
		else {
			in = d->dev_outSrc;
			out = d->dev_inSrc;
			texIn = d->texOut;
		}

		copy_kernel <<<blocks, threads>>> (d->texConstSrc, in);
		blend_kernel <<<blocks, threads>>> (texIn, out);
		dstOut = !dstOut;
	}

	float_to_color <<<blocks, threads>>> (d->output_bitmap, d->dev_inSrc);

	ErrorCheck(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(),
		cudaMemcpyDeviceToHost));

	ErrorCheck(cudaEventRecord(d->stop, 0));
	ErrorCheck(cudaEventSynchronize(d->stop));

	float elapsedTime;
	ErrorCheck(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));

	d->totalTime += elapsedTime;
	++d->frames;
	printf("Average Time per frame: %3.1f ms\n", d->totalTime / d->frames);

}

void anim_exit(DataBlock* d)
{
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);

	ErrorCheck(cudaEventDestroy(d->start));
	ErrorCheck(cudaEventDestroy(d->start));

	cudaDestroyTextureObject(d->texIn);
	cudaDestroyTextureObject(d->texOut);
	cudaDestroyTextureObject(d->texConstSrc);

}


int main()
{	
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;

	ErrorCheck(cudaEventCreate(&data.start));
	ErrorCheck(cudaEventCreate(&data.stop));

	size_t image_size = bitmap.image_size();

	ErrorCheck(cudaMalloc((void**)&data.output_bitmap, image_size));


	ErrorCheck(cudaMalloc((void**)&data.dev_inSrc, image_size));
	ErrorCheck(cudaMalloc((void**)&data.dev_outSrc, image_size));
	ErrorCheck(cudaMalloc((void**)&data.dev_constSrc, image_size));


	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = data.dev_inSrc;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.sizeInBytes = image_size;
	resDesc.res.linear.desc.x = 32;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	//cudaTextureObject_t texIn = 0, texOut = 0, texConstSrc = 0;
	cudaCreateTextureObject(&data.texIn, &resDesc, &texDesc, NULL);
	resDesc.res.linear.devPtr = data.dev_outSrc;
	cudaCreateTextureObject(&data.texOut, &resDesc, &texDesc, NULL);
	resDesc.res.linear.devPtr = data.dev_constSrc;
	cudaCreateTextureObject(&data.texConstSrc, &resDesc, &texDesc, NULL);


	float* temp = (float*)malloc(bitmap.image_size());
	for (int i = 0; i < DIM * DIM; ++i)
	{
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x > 600) && (x < 900) && (y > 700) && (y < 900)) temp[i] = MAX_TEMP;
		if ((x > 200) && (x < 350) && (y > 200) && (y < 350)) temp[i] = (MAX_TEMP+MIN_TEMP)/2;
	}

	ErrorCheck(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

	free(temp);
	
	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);

	return 0;
}
