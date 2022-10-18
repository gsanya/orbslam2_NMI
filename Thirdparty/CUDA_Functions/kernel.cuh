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

#define ENMI 0
#define SUC 1

#define MATCHING_NMI 0
#define MATCHING_HOG 1
#define MATCHING_CANNY 2
#define MATCHING_HOUGH 3


#include <opencv2/cudawarping.hpp>

#include <vector>

namespace CUDAF {
	//no masks:
	void NMIWithCuda_noMask(cv::cuda::PtrStep<unsigned char> *d_Warped,	int NMI_mode, int MatchingMode, int width, int height, float *NMI, unsigned int syntGL);
}
