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

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
#define LOG2_WARP_SIZE 5U // 5 = log (32) / log (2)
#define WARP_SIZE 32

//May change on future hardware, so better parametrize the code
#define SHARED_MEMORY_BANKS 16
#define WARP_COUNT256 16

#define HISTOGRAM256_BIN_COUNT 256
#define JOINT_HISTOGRAM256_BIN_COUNT (HISTOGRAM256_BIN_COUNT*HISTOGRAM256_BIN_COUNT)

// Threadblock size
#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT256 * WARP_SIZE)	//512
// Histogram shared memory per threadblock
#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT256 * HISTOGRAM256_BIN_COUNT)		//4096
// Joint histogram shared memory
#define JOINT_HISTOGRAM256_THREADBLOCK_MEMORY (HISTOGRAM256_BIN_COUNT * HISTOGRAM256_BIN_COUNT)	






#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

////////////////////////////////////////////////////////////////////////////////
// GPU histogram
////////////////////////////////////////////////////////////////////////////////
extern "C" void initHistogram256all(void);
extern "C" void closeHistogram256all(void);

extern "C" void histogram256all(
	uint *d_JointHistogram,
	uint *d_Histogram1,
	uint *d_Histogram2,
	uchar *d_Warped,
	uint width,
	uint height,
	cudaArray *synthCUDA
);


__global__ void ComputeEntropyKernel(uint *d_Histogram1, uint* d_Histogram2, uint* d_JointHistogram, int length, float* d_EntropyArray1, float* d_EntropyArray2, float* d_JointEntropyArray);

__global__ void AddvectorParwiseMidKernel(float* d_Array, float *d_out);

__global__ void AddVectorPairwiseKernel(float* d_Array1, float* d_Array2, float* d_Array3);