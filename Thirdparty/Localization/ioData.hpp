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

#include<opencv2/opencv.hpp>
#include<vector>

#include"cameraSettings.hpp"



////////////////////////////////////////////////////////////////////////////////////////
////																				////
////		Képek kimentésére szolgáló függvények. Részletek *.cpp-ben.				////
////																				////
////////////////////////////////////////////////////////////////////////////////////////

void setCoordinates(const char* fileName, std::vector<CameraSettings> &cameraSettings);

bool saveBMP(const char* fileName, cv::Mat& Image, std::vector<unsigned char>& synthetic);

bool saveImage(const char* fileName, cv::Mat& Image, std::vector<unsigned char>& synthetic);

bool saveImage(const char* fileName, cv::Mat& Image1, std::vector<unsigned char>& synthetic1, cv::Mat& Image2, std::vector<unsigned char>& synthetic2);

bool saveImage(const char* fileName, const int& height, const int& width, std::vector<unsigned char>& synthetic);

CameraSettings setupCam(cv::Mat &Pos, cv::Mat &Rot, cv::Mat &K_Mat, cv::Mat &transform, cv::Mat &onlyrot);

CameraSettings setupCam(cv::Mat &PosInverse, cv::Mat &K_Mat);
