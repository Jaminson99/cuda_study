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
};


texture<float> texConstSrc;
texture<float> texIn;
texture<float> texOut;


__global__ void blend_kernel(float* dst, bool dstOut)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = x + y * (blockDim.x * gridDim.x);


	int left = (x == 0) ? offset : offset-1;
	int right = (x == DIM - 1) ? offset : offset+1;

	int top = (y == 0) ? offset : offset - DIM;
	int bottom = (y == DIM - 1) ? offset : offset + DIM;

	float t, l, c, r, b;

	if (dstOut) {
		t = tex1Dfetch(texIn, top);
		l = tex1Dfetch(texIn, left);
		c = tex1Dfetch(texIn, offset);
		r = tex1Dfetch(texIn, right);
		b = tex1Dfetch(texIn, bottom);
	}
	else {
		t = tex1Dfetch(texOut, top);
		l = tex1Dfetch(texOut, left);
		c = tex1Dfetch(texOut, offset);
		r = tex1Dfetch(texOut, right);
		b = tex1Dfetch(texOut, bottom);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);

}


__global__ void copy_kernel(float* iptr)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = x + y * (blockDim.x * gridDim.x);

	float c = tex1Dfetch(texConstSrc, offset);
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

		if (dstOut) {
			in = d->dev_inSrc;
			out = d->dev_outSrc;
		}
		else {
			in = d->dev_outSrc;
			out = d->dev_inSrc;
		}

		copy_kernel<<<blocks, threads>>>(in);
		blend_kernel<<<blocks, threads>>>(out, dstOut);
		dstOut =! dstOut;
	}

	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);

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
	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConstSrc);

	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);

	ErrorCheck(cudaEventDestroy(d->start));
	ErrorCheck(cudaEventDestroy(d->start));
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

	ErrorCheck(cudaBindTexture(0, texConstSrc, data.dev_constSrc, image_size));
	ErrorCheck(cudaBindTexture(0, texIn, data.dev_inSrc, image_size));
	ErrorCheck(cudaBindTexture(0, texOut, data.dev_outSrc, image_size));


	float* temp = (float*)malloc(bitmap.image_size());
	for (int i = 0; i < DIM * DIM; ++i)
	{
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 600)) temp[i] = MAX_TEMP;
	}
	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;

	ErrorCheck(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

	free(temp);

	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);

	return 0;
}
