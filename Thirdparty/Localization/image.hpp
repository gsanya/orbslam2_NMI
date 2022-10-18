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
//Include OpenCV
#include <opencv2/core/core.hpp>

#include "opencv2/core/cuda.hpp"

class Image
{
	cv::Mat imgOriginal;
	cv::cuda::GpuMat imgOriginalGPU;
	cv::cuda::GpuMat ***warpedImagesGPU;

	int numWarpZ, numWarpY, numWarpX;
	float stepRadZ, stepRadY, stepRadX; //radian

	cv::Mat ***warpingMatrixes;

	cv::Mat K_mat;
	
	int width, height;

public:

	Image(const int &numWarpz, const int &numWarpy, const int &numWarpx, const float &stepWarpz, const float &stepWarpy, const float &stepWarpx, const int &width, const int &height, cv::Mat K);

	void calculateWarping();

	void loadOriginal(cv::Mat Original);

	void resizeKernel(const int &numwarpx, const int &numwarpy, const int &numwarpz, const float &stepradx, const float &steprady, const float &stepradz);

	cv::Mat getOriginal();

	cv::cuda::GpuMat getImageGPU(const int &indexZ, const int &indexY, const int &indexX);

	cv::Mat getK();
	
	~Image();

	float getStepZ();
	float getStepY();
	float getStepX();

	void setStepZ(float sz);
	void setStepY(float sy);
	void setStepX(float sx);

	int getNumWarpZ();
	int getNumWarpY();
	int getNumWarpX();

	void setNumWarpZ(int wz);
	void setNumWarpY(int wy);
	void setNumWarpX(int wx);
};