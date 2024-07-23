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

	float t, l, c, r, b, tl, tr, bl, br;

	t = tex2D<float>(in, x, y - 1);
	l = tex2D<float>(in, x - 1, y);
	c = tex2D<float>(in, x, y);
	r = tex2D<float>(in, x + 1, y);
	b = tex2D<float>(in, x, y + 1);

	tl = tex2D<float>(in, x - 1, y - 1);
	tr = tex2D<float>(in, x + 1, y - 1);
	bl = tex2D<float>(in, x - 1, y + 1);
	br = tex2D<float>(in, x + 1, y + 1);


	dst[offset] = c + SPEED * (t + b + r + l - 4 * c) / 2 + SPEED * (tl + tr + br + bl - 4 * c) / 2.42;
}


__global__ void copy_kernel(cudaTextureObject_t texConstSrc, float* iptr)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = x + y * (blockDim.x * gridDim.x);

	float c = tex2D<float>(texConstSrc, x, y);
	if (c != 0) iptr[offset] = c;
}


void anim_gpu(DataBlock* d, int ticks)
{
	ErrorCheck(cudaEventRecord(d->start, 0));

	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	CPUAnimBitmap* bitmap = d->bitmap;

	volatile bool dstOut = true;
	//copy_kernel <<<blocks, threads >>> (d->texConstSrc, d->dev_inSrc);
	for (int i = 0; i < 90; ++i)
	{
		float* in, * out;
		cudaTextureObject_t texIn;

		if (dstOut) {
			out = d->dev_outSrc;
			texIn = d->texIn;
		}
		else {
			out = d->dev_inSrc;
			texIn = d->texOut;
		}

		//copy_kernel <<<blocks, threads >>> (d->texConstSrc, in);
		blend_kernel <<<blocks, threads >>> (texIn, out);
		dstOut = !dstOut;
	}

	float_to_color <<<blocks, threads >>> (d->output_bitmap, d->dev_inSrc);

	ErrorCheck(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(),
		cudaMemcpyDeviceToHost));

	ErrorCheck(cudaEventRecord(d->stop, 0));
	ErrorCheck(cudaEventSynchronize(d->stop));

	float elapsedTime;
	ErrorCheck(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));

	d->totalTime += elapsedTime;
	++d->frames;
	printf("Average Time per frame: %3.2f ms\n", d->totalTime / d->frames);

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
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = data.dev_inSrc;
	resDesc.res.pitch2D.width = DIM;
	resDesc.res.pitch2D.height = DIM;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
	resDesc.res.pitch2D.pitchInBytes = image_size / DIM;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&data.texIn, &resDesc, &texDesc, NULL);
	resDesc.res.pitch2D.devPtr = data.dev_outSrc;
	cudaCreateTextureObject(&data.texOut, &resDesc, &texDesc, NULL);
	resDesc.res.pitch2D.devPtr = data.dev_constSrc;
	cudaCreateTextureObject(&data.texConstSrc, &resDesc, &texDesc, NULL);


	float* temp = (float*)malloc(bitmap.image_size());
	for (int i = 0; i < DIM * DIM; ++i)
	{
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x > 400) && (x < 700) && (y > 600) && (y < 900)) temp[i] = MAX_TEMP;
		if ((x > 300) && (x < 500) && (y > 300) && (y < 500)) temp[i] = MAX_TEMP;
	}


	ErrorCheck(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

	free(temp);

	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	copy_kernel <<<blocks, threads>>> (data.texConstSrc, data.dev_inSrc);
	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);

	return 0;
}
