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
// Include GLM
#include <glm/glm.hpp>

////////////////////////////////////////////////////////////////////////////////////////////
////																					////
////		Gyakorlatilag csak egy struct rejtett változókkal és get függvényekkel		////
////																					////
////////////////////////////////////////////////////////////////////////////////////////////

class CameraSettings
{
	std::string fileName;
	cv::Mat K_Matrix;
	glm::vec3 Camera_position;
	glm::vec3 Camera_direction;
	glm::vec3 Camera_up;
public:
	CameraSettings(glm::vec3 Pos, glm::vec3 Dir, glm::vec3 Up) :Camera_position(Pos), Camera_direction(Dir), Camera_up(Up) {}
	CameraSettings(std::string file, cv::Mat K, glm::vec3 Pos, glm::vec3 Dir, glm::vec3 Up) :fileName(file), K_Matrix(K), Camera_position(Pos), Camera_direction(Dir), Camera_up(Up) {}
	glm::vec3 getPosition() { return Camera_position; }
	glm::vec3 getDirection() { return Camera_direction; }
	glm::vec3 getUp() { return Camera_up; }
	cv::Mat getK() { return K_Matrix; }
	std::string getFileName() { return fileName; }
};