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
#include "image.hpp"

#include <omp.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudawarping.hpp>

#include "allProperties.hpp"
#include "helperFunctions.hpp"


Image::Image(const int &numWarpz, const int &numWarpy, const int &numWarpx, const float &stepWarpz, const float &stepWarpy, const float &stepWarpx, const int &width, const int &height, cv::Mat K) :
	numWarpZ(numWarpz),
	stepRadZ(stepWarpz),
	numWarpY(numWarpy),
	stepRadY(stepWarpy),
	numWarpX(numWarpx),
	stepRadX(stepWarpx),
	width(width),
	height(height)
{
	K_mat = K.clone();

	//create warping matricies
	warpingMatrixes = new cv::Mat**[numWarpZ];
	for (int z = 0; z < numWarpZ; z++)
	{
		warpingMatrixes[z] = new cv::Mat*[numWarpY];
		for (int y = 0; y < numWarpY; y++)
		{
			warpingMatrixes[z][y] = new cv::Mat[numWarpX];
		}
	}
	


	warpedImagesGPU = new cv::cuda::GpuMat**[numWarpZ];		
	for (int z = 0; z < numWarpZ; z++)
	{
		warpedImagesGPU[z] = new cv::cuda::GpuMat*[numWarpY];
		for (int y = 0; y < numWarpY; y++)
		{
			warpedImagesGPU[z][y] = new cv::cuda::GpuMat[numWarpX];
			for (int x = 0; x < numWarpX; x++)
			{
				cv::cuda::createContinuous(height, width, CV_8UC1, warpedImagesGPU[z][y][x]);
			}
		}
	}
	std::cout << std::endl;
	
	//cv::cuda::createContinuous(height, width, CV_8UC1, imgOriginalGPU);

	//calculate warping matricies
	cv::Mat_<double> R, Rz, Ry, Rx;
	double thetaz = -(numWarpZ - 1) / 2 * stepRadZ;
	for (int i = 0; i < numWarpZ; i++, thetaz += stepRadZ)
	{
		//Rotation about z (points backwards)
		Rz = (cv::Mat_<double>(3, 3) <<
			cos(thetaz), -sin(thetaz), 0,
			sin(thetaz), cos(thetaz), 0,
			0, 0, 1);

		double thetay = -(numWarpY - 1) / 2 * stepRadY;
		for (int j = 0; j < numWarpY; j++, thetay += stepRadY)
		{
			//Rotation about y (point up)
			Ry = (cv::Mat_<double>(3, 3) <<
				cos(thetay), 0, sin(thetay),
				0, 1, 0,
				-sin(thetay), 0, cos(thetay));

			double thetax = -(numWarpX - 1) / 2 * stepRadX;
			for (int k = 0; k < numWarpX; k++, thetax += stepRadX)
			{
				//Rotation about x (points right)
				Rx = (cv::Mat_<double>(3, 3) <<
					1, 0, 0,
					0, cos(thetax), -sin(thetax),
					0, sin(thetax), cos(thetax));

				R = Rz * Ry*Rx; //first rotate about x, then about y, and then about z
				warpingMatrixes[i][j][k] = K * R * K.inv();
			}
		}
	}



}

//calculates the warped images
void Image::calculateWarping()
{
	for (int i = 0; i < numWarpZ; i++)
	{
		for (int j = 0; j < numWarpY; j++)
		{
			for (int k = 0; k < numWarpX; k++)
			{
				cv::cuda::warpPerspective(imgOriginalGPU, warpedImagesGPU[i][j][k], warpingMatrixes[i][j][k], imgOriginal.size());
			}
		}
	}
	
}

void Image::loadOriginal(cv::Mat Original)
{
	imgOriginalGPU.release();
	Original.copyTo(imgOriginal);
	imgOriginalGPU.upload(imgOriginal);  
}

cv::Mat Image::getOriginal()
{
	return imgOriginal;
}

cv::cuda::GpuMat Image::getImageGPU(const int &indexZ, const int &indexY, const int &indexX)
{
	return warpedImagesGPU[indexZ][indexY][indexX];
}

cv::Mat Image::getK()
{
	return K_mat;
}

Image::~Image()
{

	for (int z = 0; z < numWarpZ; z++)
	{
		for (int y = 0; y < numWarpY; y++)
		{
			for (int x = 0; x < numWarpX; x++)
			{
				warpedImagesGPU[z][y][x].release();
			}
			delete[] warpedImagesGPU[z][y];
			delete[] warpingMatrixes[z][y];
		}
		delete[] warpedImagesGPU[z];
		delete[] warpingMatrixes[z];
	}
	delete[] warpedImagesGPU;
	delete[] warpingMatrixes;
	
	imgOriginalGPU.release();

}

void Image::resizeKernel(const int &numwarpx, const int &numwarpy, const int &numwarpz, const float &stepradx, const float &steprady, const float &stepradz) 
{
	for (int z = 0; z < numWarpZ; z++)
	{
		for (int y = 0; y < numWarpY; y++)
		{
			for (int x = 0; x < numWarpX; x++)
			{
				warpedImagesGPU[z][y][x].release();
			}
			delete[] warpedImagesGPU[z][y];
			delete[] warpingMatrixes[z][y];
		}
		delete[] warpedImagesGPU[z];
		delete[] warpingMatrixes[z];
	}

	delete[] warpedImagesGPU;
	delete[] warpingMatrixes;

	imgOriginalGPU.release();

	setStepX(stepradx);
	setStepY(steprady);
	setStepZ(stepradz);

	setNumWarpX(numwarpx);
	setNumWarpY(numwarpy);
	setNumWarpZ(numwarpz);

	//create warped Images matricies
	warpedImagesGPU = new cv::cuda::GpuMat**[numWarpZ];

	for (int z = 0; z < numWarpZ; z++)
	{
		warpedImagesGPU[z] = new cv::cuda::GpuMat*[numWarpY];
		for (int y = 0; y < numWarpY; y++)
		{
			warpedImagesGPU[z][y] = new cv::cuda::GpuMat[numWarpX];
			for (int x = 0; x < numWarpX; x++)
			{
				cv::cuda::createContinuous(height, width, CV_8UC1, warpedImagesGPU[z][y][x]);
			}
		}
	}
	std::cout << std::endl;
	//cv::cuda::createContinuous(height, width, CV_8UC1, imgOriginalGPU);

	//create warping matricies
	warpingMatrixes = new cv::Mat**[numWarpZ];
	for (int z = 0; z < numWarpZ; z++)
	{
		warpingMatrixes[z] = new cv::Mat*[numWarpY];
		for (int y = 0; y < numWarpY; y++)
		{
			warpingMatrixes[z][y] = new cv::Mat[numWarpX];
		}
	}

	//calculate warping matricies
	cv::Mat_<double> R, Rz, Ry, Rx;
	double thetaz = -(numWarpZ - 1) / 2 * stepRadZ;
	for (int i = 0; i < numWarpZ; i++, thetaz += stepRadZ)
	{
		//Rotation about z (points backwards)
		Rz = (cv::Mat_<double>(3, 3) <<
			cos(thetaz), -sin(thetaz), 0,
			sin(thetaz), cos(thetaz), 0,
			0, 0, 1);

		double thetay = -(numWarpY - 1) / 2 * stepRadY;
		for (int j = 0; j < numWarpY; j++, thetay += stepRadY)
		{
			//Rotation about y (point up)
			Ry = (cv::Mat_<double>(3, 3) <<
				cos(thetay), 0, sin(thetay),
				0, 1, 0,
				-sin(thetay), 0, cos(thetay));

			double thetax = -(numWarpX - 1) / 2 * stepRadX;
			for (int k = 0; k < numWarpX; k++, thetax += stepRadX)
			{
				//Rotation about x (points right)
				Rx = (cv::Mat_<double>(3, 3) <<
					1, 0, 0,
					0, cos(thetax), -sin(thetax),
					0, sin(thetax), cos(thetax));

				R = Rz * Ry*Rx; //first rotate about x, then about y, and then about z
				warpingMatrixes[i][j][k] = K_mat * R * K_mat.inv();
			}
		}
	}	
}

float Image::getStepZ() { return stepRadZ; }
float Image::getStepY() { return stepRadY; }
float Image::getStepX() { return stepRadX; }

int Image::getNumWarpZ() { return numWarpZ; }
int Image::getNumWarpY() { return numWarpY; }
int Image::getNumWarpX() { return numWarpX; }

void Image::setStepZ(float sz) { stepRadZ=sz; }
void Image::setStepY(float sy) { stepRadY=sy; }
void Image::setStepX(float sx) { stepRadX=sx; }

void Image::setNumWarpZ(int wz) {  numWarpZ=wz; }
void Image::setNumWarpY(int wy) {  numWarpY=wy; }
void Image::setNumWarpX( int wx) {  numWarpX=wx; }