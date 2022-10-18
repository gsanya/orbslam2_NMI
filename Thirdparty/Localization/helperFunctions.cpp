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
#include "helperFunctions.hpp"

//Include OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

//basic cpp
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <windows.h>
#include <Shellapi.h>
#include <ctime>

//CUDA
#include "kernel.cuh"

//headers
#include "nmiSearchKernel.hpp"
#include "allProperties.hpp"
#include "iodata.hpp"

//overload of the above function. I use this
std::vector<NmiSearchKernel> helperFunctions::find_max_elements(float ******nmi, NmiSearchKernel &nmiKernel)
{
	float max = 0;
	for (int wz = 0; wz < nmiKernel.getNumWarpZ(); wz++)
	{
		for (int wy = 0; wy < nmiKernel.getNumWarpY(); wy++)
		{
			for (int wx = 0; wx < nmiKernel.getNumWarpX(); wx++)
			{
				for (int sz = 0; sz < nmiKernel.getNumSynthZ(); sz++)
				{
					for (int sy = 0; sy < nmiKernel.getNumSynthY(); sy++)
					{
						for (int sx = 0; sx < nmiKernel.getNumSynthX(); sx++)
						{
							if (nmi[wz][wy][wx][sz][sy][sx] > max)
							{
								max = nmi[wz][wy][wx][sz][sy][sx];
							}
						}
					}
				}
			}
		}
	}

	std::vector<NmiSearchKernel> maxElements;
	for (int wz = 0; wz < nmiKernel.getNumWarpZ(); wz++)
	{
		for (int wy = 0; wy < nmiKernel.getNumWarpY(); wy++)
		{
			for (int wx = 0; wx < nmiKernel.getNumWarpX(); wx++)
			{
				for (int sz = 0; sz < nmiKernel.getNumSynthZ(); sz++)
				{
					for (int sy = 0; sy < nmiKernel.getNumSynthY(); sy++)
					{
						for (int sx = 0; sx < nmiKernel.getNumSynthX(); sx++)
						{
							if (nmi[wz][wy][wx][sz][sy][sx] == max)
							{

								NmiSearchKernel maxElem = NmiSearchKernel();
								maxElem.setBest(sx, sy, sz, wx, wy, wz, nmi[wz][wy][wx][sz][sy][sx]);
								maxElements.push_back(maxElem);
							}
						}
					}
				}
			}
		}
	}
	return maxElements;
}

void helperFunctions::log(std::stringstream &text, std::string Filename)
{
	std::ofstream logout;
	logout.open(Filename, std::ios_base::app);
	std::cout << text.str();
	logout << text.str();
	logout.close();
	text.str("");
}