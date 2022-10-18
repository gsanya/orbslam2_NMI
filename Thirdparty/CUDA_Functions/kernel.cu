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
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>
#include <iostream>
#include <chrono>
#include <stdio.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "NMI.cuh"
#include "kernel.cuh"




namespace CUDAF {
	void NMIWithCuda_noMask(cv::cuda::PtrStep<unsigned char> *d_data2GPU, int NMI_mode,  int MatchingMode, int width, int height, float *NMI, unsigned int syntGL)
	{
		//Map openGL texture to cuda array		
		cudaGraphicsResource_t resources[1] = {0};
		checkCudaErrors(cudaGraphicsGLRegisterImage(&resources[0], syntGL, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
		checkCudaErrors(cudaGraphicsMapResources(1, resources, 0));
		cudaArray *synthCUDA;
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&synthCUDA, resources[0], 0, 0));
		
		//GPU memory is with d_(device) prefix, CPU memory is with h_(host) prefix
		uint  *d_Histogram1, *d_Histogram2, *d_JointHistogram;
		float * d_Entropy1, *d_Entropy2, *d_JointEntropy, *d_JointEntropyShort;

		//for nice look
		dim3 blocksPerGrid(0, 0, 0);
		dim3 threadsPerBlock(0, 0, 0);
		//asd

		checkCudaErrors(cudaMalloc((void **)&d_Histogram1, 256 * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&d_Histogram2, 256 * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&d_JointHistogram, 256 * 256 * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&d_Entropy1, 256 * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_Entropy2, 256 * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_JointEntropyShort, 256 * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_JointEntropy, 256 * 256 * sizeof(float)));
		

		initHistogram256all();

		//calculates histogram and joint histogram
		histogram256all(d_JointHistogram, d_Histogram1, d_Histogram2, (uchar*)d_data2GPU, width, height, synthCUDA);													
	

		//computes "entropy" for each histogram bin
		blocksPerGrid = { 258, 1, 1 };
		threadsPerBlock = { 256, 1, 1 }; 
		ComputeEntropyKernel << < blocksPerGrid, threadsPerBlock >> > (d_Histogram1, d_Histogram2, d_JointHistogram, width*height, d_Entropy1, d_Entropy2, d_JointEntropy);	


		blocksPerGrid = { 256,1,1 };
		threadsPerBlock = { 128,1,1 };
		AddvectorParwiseMidKernel << < blocksPerGrid, threadsPerBlock >> > (d_JointEntropy, d_JointEntropyShort);

		//calculates the entropies, and than the nmi to d_Entropy1[0]
		blocksPerGrid = { 3,1,1 };
		threadsPerBlock = { 128,1,1 };
		AddVectorPairwiseKernel << < blocksPerGrid, threadsPerBlock >> > (d_Entropy1, d_Entropy2, d_JointEntropyShort);
					
		closeHistogram256all();				

		//copy the calculated NMI to the CPU
		checkCudaErrors(cudaMemcpy((void*)&NMI[0], d_Entropy1,  sizeof(float), cudaMemcpyDeviceToHost));

		//deletes
		checkCudaErrors(cudaFree(d_Histogram1));
		checkCudaErrors(cudaFree(d_Histogram2));
		checkCudaErrors(cudaFree(d_JointHistogram));
		checkCudaErrors(cudaFree(d_Entropy1));
		checkCudaErrors(cudaFree(d_Entropy2));
		checkCudaErrors(cudaFree(d_JointEntropyShort));
		checkCudaErrors(cudaFree(d_JointEntropy));

		//unmap openGL resources
		checkCudaErrors(cudaGraphicsUnmapResources(1,resources));
		checkCudaErrors(cudaGraphicsUnregisterResource(resources[0]));
	}
}