/**
* This file is part of orbslam2_NMI.
*
* Copyright (C) 2021 Sándor Gazdag <gazdag.sandor at sztaki dot hu> (SZTAKI)
* For more information see <https://github.com/gsanya/orbslam2_NMI>
*
* orbslam2_NMI is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* orbslam2_NMI is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "NMI.cuh"
#include "kernel.cuh"
#include "allProperties.hpp"

//texture for opengl rendered texture
texture<uchar, cudaTextureType2D, cudaReadModeElementType> texCUDA;

//atomics
inline __device__ void addByte_noBG(uint tid, uint *d_PartialJointHistograms, uint *s_WarpHist1, uint *s_WarpHist2, uint data1, uint data2)
{
	uint d1 = data1;
	uint d2 = data2;
	atomicAdd(s_WarpHist1 + d1, 1); //atomicAdd(memloc, value to add)
	atomicAdd(s_WarpHist2 + d2, 1);
	atomicAdd(d_PartialJointHistograms + (tid >> LOG2_WARP_SIZE) * JOINT_HISTOGRAM256_BIN_COUNT + d1 * HISTOGRAM256_BIN_COUNT + d2, 1);
}


__global__ void histogram256Kernel(
	uint *d_PartialJointHistograms,
	uint *d_PartialHistograms1,
	uint *d_PartialHistograms2,
	uchar *d_Warped,
	uint width,
	uint height)
{
	//shared histograms (one for every warp)
	__shared__ uint s_Hist1[HISTOGRAM256_THREADBLOCK_MEMORY]; //  warp_count * 256 = 16*256 = 4096
	__shared__ uint s_Hist2[HISTOGRAM256_THREADBLOCK_MEMORY]; //  warp_count * 256 --> összesen 8192*4= 32768 Byte= 32 kByte shared memory kéne blokkonként (nvidia Geforce 950M-nek 65536 Byte, szval elég)


	// calculating starting position for the warp
	uint *s_WarpHist1 = s_Hist1 + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;
	uint *s_WarpHist2 = s_Hist2 + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;


	// Clear shared memory storage for current threadblock before processing
#pragma unroll
	for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
	{
		s_Hist1[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
		s_Hist2[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
	}
	__syncthreads();	//syncs threads in block

	for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < width*height; pos += UMUL(blockDim.x, gridDim.x))
	{

		uint data1 = uint(tex2D(texCUDA, float(pos % width), float(height - 1) - float(pos / width)));
		uint data2 = uint(d_Warped[pos]);

		if (nmi_prop_BG || ((data1 != 0) && (data2 != 0)))
			addByte_noBG(threadIdx.x, d_PartialJointHistograms, s_WarpHist1, s_WarpHist2, data1, data2);
	}
	__syncthreads();

	//Merge per-warp histograms into per-block and write to global memory
	for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
	{
		uint sum1 = 0;
		uint sum2 = 0;
		for (uint i = 0; i < WARP_COUNT256; i++)
		{
			sum1 += s_Hist1[bin + i * HISTOGRAM256_BIN_COUNT];
			sum2 += s_Hist2[bin + i * HISTOGRAM256_BIN_COUNT];
		}
		// per block sub-histogram 
		d_PartialHistograms1[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum1;
		d_PartialHistograms2[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum2;
	}
}


#define MERGE_THREADBLOCK_SIZE 1024


__global__ void mergeHistogram256Kernel(
	uint *d_Histogram1,
	uint *d_Histogram2,
	uint *d_PartialHistograms1,
	uint *d_PartialHistograms2,
	uint histogramCount)
{
	uint sum1 = 0;
	uint sum2 = 0;
	

	for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)	
	{
		sum1 += d_PartialHistograms1[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];
		sum2 += d_PartialHistograms2[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];
	}

	//shared only inside blocks
	__shared__ uint data1[MERGE_THREADBLOCK_SIZE];
	__shared__ uint data2[MERGE_THREADBLOCK_SIZE];

	data1[threadIdx.x] = sum1;
	data2[threadIdx.x] = sum2;
	for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (threadIdx.x < stride)
		{
			data1[threadIdx.x] += data1[threadIdx.x + stride];
			data2[threadIdx.x] += data2[threadIdx.x + stride];
		}
	}
	if (threadIdx.x == 0)
	{
		d_Histogram1[blockIdx.x] = data1[0];
		d_Histogram2[blockIdx.x] = data2[0];
	}
}

__global__ void mergeJointHistogram256Kernel(
	uint *d_JointHistogram,
	uint *d_PartialHistograms,
	uint jointHistogramCount)
{
	uint sum = 0;
#pragma unroll
	for (int i = 0; i < jointHistogramCount; i++)
	{
		sum += d_PartialHistograms[blockIdx.x*blockDim.x + threadIdx.x + i * JOINT_HISTOGRAM256_BIN_COUNT];
	}
	d_JointHistogram[blockIdx.x*blockDim.x + threadIdx.x] = sum;
}


static const uint  PARTIAL_HISTOGRAM_COUNT = 240;
static uint        *d_PartialHistograms1;
static uint        *d_PartialHistograms2;
static uint        *d_PartialJointHistograms;


//Internal memory allocation
extern "C" void initHistogram256all(void)
{
	checkCudaErrors(cudaMalloc((void **)&d_PartialHistograms1, PARTIAL_HISTOGRAM_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&d_PartialHistograms2, PARTIAL_HISTOGRAM_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&d_PartialJointHistograms, WARP_COUNT256 * JOINT_HISTOGRAM256_BIN_COUNT * sizeof(uint)));

}

//Internal memory deallocation
extern "C" void closeHistogram256all(void)
{
	checkCudaErrors(cudaFree(d_PartialHistograms1));
	checkCudaErrors(cudaFree(d_PartialHistograms2));
	checkCudaErrors(cudaFree(d_PartialJointHistograms));
}

// wrapper function
extern "C" void histogram256all(
	uint *d_JointHistogram,
	uint *d_Histogram1,
	uint *d_Histogram2,
	uchar *d_Warped,
	uint width,
	uint height,
	cudaArray *synthCUDA
)
{
	checkCudaErrors(cudaMemset(d_PartialJointHistograms, 0, WARP_COUNT256 * JOINT_HISTOGRAM256_BIN_COUNT * sizeof(uint)));

	checkCudaErrors(cudaBindTextureToArray(texCUDA, synthCUDA));

	//<<<240,512>>>
	histogram256Kernel << <PARTIAL_HISTOGRAM_COUNT, HISTOGRAM256_THREADBLOCK_SIZE >> > (
		d_PartialJointHistograms,
		d_PartialHistograms1,
		d_PartialHistograms2,
		d_Warped,
		width,
		height);

	//<<<256,1024>>>
	mergeHistogram256Kernel << <HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE >> > (
		d_Histogram1,
		d_Histogram2,
		d_PartialHistograms1,
		d_PartialHistograms2,
		PARTIAL_HISTOGRAM_COUNT);

	// <<< 256 , 256 >>>
	mergeJointHistogram256Kernel << < HISTOGRAM256_BIN_COUNT, HISTOGRAM256_BIN_COUNT >> > (
		d_JointHistogram,
		d_PartialJointHistograms,
		WARP_COUNT256);

	checkCudaErrors(cudaUnbindTexture(texCUDA));
}


//<<<258,256>>
__global__ void ComputeEntropyKernel(
	uint *d_Histogram1,
	uint* d_Histogram2,
	uint *d_JointHistogram,
	int length,
	float* d_EntropyArray1,
	float* d_EntropyArray2,
	float* d_JointEntropyArray)
{

	if (blockIdx.x == 0)
	{
		if (d_Histogram1[threadIdx.x] == 0)
			d_EntropyArray1[threadIdx.x] = 0;
		else
			d_EntropyArray1[threadIdx.x] = ((float)d_Histogram1[threadIdx.x] / (float)length)		*		log2f((float)d_Histogram1[threadIdx.x] / (float)length);
	}
	else
	{
		if (blockIdx.x == 1)
		{
			if (d_Histogram2[threadIdx.x] == 0)
				d_EntropyArray2[threadIdx.x] = 0;
			else
				d_EntropyArray2[threadIdx.x] = ((float)d_Histogram2[threadIdx.x] / (float)length)		*		log2f((float)d_Histogram2[threadIdx.x] / (float)length);
		}
		else
		{
			if (d_JointHistogram[blockDim.x*(blockIdx.x - 2) + threadIdx.x] == 0)
				d_JointEntropyArray[blockDim.x*(blockIdx.x - 2) + threadIdx.x] = 0;
			else
			{
				int ind = blockDim.x*(blockIdx.x - 2) + threadIdx.x;
				d_JointEntropyArray[ind] = ((float)d_JointHistogram[ind] / (float)length)		 *			log2f((float)d_JointHistogram[ind] / (float)length);
			}
		}
	}
}

//<<<256, 128>>>
__global__ void AddvectorParwiseMidKernel(
	float* d_Array,
	float *d_out)
{
	int t = blockDim.x*blockIdx.x * 2 + threadIdx.x;
	int n = blockDim.x;
	while (n >= 1)
	{
		if (t - blockDim.x*blockIdx.x * 2 < n)
		{
			d_Array[t] += d_Array[t + n];
		}
		__syncthreads();
		n /= 2;
	}
	if (threadIdx.x == 0)
		d_out[blockIdx.x] = d_Array[blockDim.x*blockIdx.x * 2];
}


__global__ void AddVectorPairwiseKernel(
	float* d_Array1,
	float* d_Array2,
	float* d_Array3)
{
	if (blockIdx.x == 0)//summing Ha
	{
		int t = threadIdx.x;
		int n = blockDim.x;
		while (n != 0)
		{
			if (t < n)
			{
				d_Array1[t] += d_Array1[t + n];
			}
			__syncthreads();
			n /= 2;
		}
	}
	else
	{
		if (blockIdx.x == 1)//summing Hb
		{
			int t = threadIdx.x;
			int n = blockDim.x;
			while (n != 0)
			{
				if (t < n)
				{
					d_Array2[t] += d_Array2[t + n];
				}
				__syncthreads();
				n /= 2;
			}
		}
		else//summing Hab
		{
			int t = threadIdx.x;
			int n = blockDim.x;
			while (n != 0)
			{
				if (t < n)
				{
					d_Array3[t] += d_Array3[t + n];
				}
				__syncthreads();
				n /= 2;
			}
		}
	}
	//__syncthreads();
	//calculate NMI into d_Array1[0] on the first thread
	if (blockIdx.x == 1 && threadIdx.x == 0)
	{
		if (ENMI) {
			if (d_Array1[0] == 0 && d_Array2[0] == 0 && d_Array3[0] == 0) {
				d_Array1[0] = 0;
			}
			else {
				d_Array1[0] = ((-d_Array1[0])+(-d_Array2[0]))/(-d_Array3[0]);
			}			
		}
		else if(SUC){	
			if (d_Array1[0] == 0 && d_Array2[0] == 0 && d_Array3[0] == 0) {
				d_Array1[0] = 0;
			}
			else {
				d_Array1[0] = 2*(1-((-d_Array3[0])/((-d_Array1[0]) + (-d_Array2[0]))));
			}
		}
		else
			d_Array1[0] = -1;
	}
}



