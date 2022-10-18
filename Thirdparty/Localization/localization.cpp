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

//Include cpp and standards
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <windows.h>
#include <Shellapi.h>
#include <ctime>
#include <iostream>
#include <filesystem>
// Include GLEW
#include <GL/glew.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
//using namespace glm;

//Include Headers
#include "shader.hpp"
#include "texture.hpp"
#include "objLoader.hpp""
#include "ioData.hpp"
#include "helperFunctions.hpp"
#include "image.hpp"
#include "rendering.hpp"
#include "localization.hpp"
#include "kernel.cuh"
#include "image.hpp"
#include "rendering.hpp"

// Include GLFW
#include <glfw3.h>
GLFWwindow* window;

//Include Boost
#include <boost/filesystem.hpp>
#include <omp.h>

//Include OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

//CUDA


//TESZTHEZ
#include <chrono>

#define M_PI 3.14159265358979323846

#define MATCHING_NMI 0


NmiObjects::NmiObjects(const std::string &strSettingsFile)
{
	//create results folder
	std::stringstream folder;
	std::stringstream ss_log;
	time_t t = time(NULL);
	struct tm  curtime = *localtime(&t);

	std::stringstream sout;
	
	
	sout  << nmi_prop_OUTPUT_LOC"/" << std::put_time(&curtime, "%d-%m-%Y_%Hh%Mm%Ss");
	std::string results_path = sout.str();
	boost::filesystem::path dir(results_path.c_str());

	if (!dir.is_absolute()) {
		boost::filesystem::path working_dir(boost::filesystem::current_path());
		sout.str("");
		sout << working_dir.string() <<nmi_prop_OUTPUT_LOC"/" << std::put_time(&curtime, "%d-%m-%Y_%Hh%Mm%Ss");
		std::cout << sout.str();
		dir = boost::filesystem::path(sout.str().c_str());
	}
	try
	{
		if (boost::filesystem::create_directory(dir)) {
			ss_log << "new directory: ";
		}
		else
		{
			ss_log << "existing directory: ";
		}
		resultsPath = dir.string();
	}
	catch (...)
	{
		ss_log << "Unable to find the right directory!" << std::endl;
	}
	ss_log << resultsPath << std::endl;
	logPath = resultsPath + "/_log.txt";
	ss_log << "logPath: " << logPath << std::endl;
	helperFunctions::log(ss_log, logPath);
	N = 0;
	
	
	//only initvalues
	glm::vec3 Camera_position = { 0.0f,0.0f,0.0f };
	glm::vec3 Camera_direction = { 0.0f,0.0f,0.0f };
	glm::vec3 Camera_up = { 0.0f,0.0f,0.0f };

	cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);
	//MyRenderer is created
	myRenderer = new Rendering<nmi_prop_RENDER>(
		fSettings["NMI.Render.PointSize"],
		1280,
		720,
		fSettings["Camera.Width"], 
		fSettings["Camera.Height"],
		fSettings["NMI.SynthNumX"],
		fSettings["NMI.SynthNumY"],
		fSettings["NMI.SynthNumZ"],
		fSettings["NMI.SynthStepX"], 
		fSettings["NMI.SynthStepY"],
		fSettings["NMI.SynthStepZ"],
		Camera_position,
		Camera_direction,
		Camera_up,
		fSettings["NMI.Render.NearPlane"],
		fSettings["NMI.Render.FarPlane"],
		fSettings["Camera.fx"],
		fSettings["Camera.fy"],
		fSettings["Camera.cx"],
		fSettings["Camera.cy"],
		fSettings["NMI.Render.Object"],
		fSettings["NMI.Render.Texture"],
		fSettings["NMI.Render.Cloud"],
		fSettings["NMI.Render.Offset"],
		logPath);

	double fx = fSettings["Camera.fx"];
	double fy = fSettings["Camera.fy"];
	double cx = fSettings["Camera.cx"];
	double cy = fSettings["Camera.cy"];

	cv::Mat_<double> K = cv::Mat::eye(3, 3, CV_32F);
	K.at<double>(0, 0) = fx;
	K.at<double>(1, 1) = fy;
	K.at<double>(0, 2) = cx;
	K.at<double>(1, 2) = cy;

	//MyImage is created
	myImage = new Image(
		fSettings["NMI.WarpNumZ"],
		fSettings["NMI.WarpNumY"],
		fSettings["NMI.WarpNumX"],
		fSettings["NMI.WarpStepZ"],
		fSettings["NMI.WarpStepY"],
		fSettings["NMI.WarpStepX"],
		fSettings["Camera.Width"],
		fSettings["Camera.Height"],
		K);
	

	//rating is created
	rating = new float*****[myImage->getNumWarpZ()];
	omp_set_num_threads(nmi_prop_NUMBER_OF_THREADS);
#pragma omp parallel
	{
#pragma omp for schedule(static,1)
		for (int wz = 0; wz < myImage->getNumWarpZ(); wz++)
		{
			rating[wz] = new float****[myImage->getNumWarpY()];
			for (int wy = 0; wy < myImage->getNumWarpY(); wy++)
			{
				rating[wz][wy] = new float***[myImage->getNumWarpX()];
				for (int wx = 0; wx < myImage->getNumWarpX(); wx++)
				{
					rating[wz][wy][wx] = new float**[myRenderer->getNumSynthZ()];
					for (int sz = 0; sz < myRenderer->getNumSynthZ(); sz++)
					{
						rating[wz][wy][wx][sz] = new float*[myRenderer->getNumSynthY()];
						for (int sy = 0; sy < myRenderer->getNumSynthY(); sy++)
						{
							rating[wz][wy][wx][sz][sy] = new float[myRenderer->getNumSynthX()];
						}
					}
				}
			}
		}
	}///parallel end

	

	//initialize the kernel objects
	NmiKernel = new NmiSearchKernel(
		fSettings["NMI.SynthNumX"],
		fSettings["NMI.SynthNumY"],
		fSettings["NMI.SynthNumZ"],
		fSettings["NMI.WarpNumX"],
		fSettings["NMI.WarpNumY"],
		fSettings["NMI.WarpNumZ"],
		fSettings["NMI.SynthStepX"],
		fSettings["NMI.SynthStepY"],
		fSettings["NMI.SynthStepZ"],
		fSettings["NMI.WarpStepX"],
		fSettings["NMI.WarpStepY"],
		fSettings["NMI.WarpStepZ"]);
	LastNmiKernel = new NmiSearchKernel(
		fSettings["NMI.SynthNumX"],
		fSettings["NMI.SynthNumY"],
		fSettings["NMI.SynthNumZ"],
		fSettings["NMI.WarpNumX"],
		fSettings["NMI.WarpNumY"],
		fSettings["NMI.WarpNumZ"],
		fSettings["NMI.SynthStepX"],
		fSettings["NMI.SynthStepY"],
		fSettings["NMI.SynthStepZ"],
		fSettings["NMI.WarpStepX"],
		fSettings["NMI.WarpStepY"],
		fSettings["NMI.WarpStepZ"]);
	InitialNmiKernel = new NmiSearchKernel(
		fSettings["NMI.SynthNumX"],
		fSettings["NMI.SynthNumY"],
		fSettings["NMI.SynthNumZ"],
		fSettings["NMI.WarpNumX"],
		fSettings["NMI.WarpNumY"],
		fSettings["NMI.WarpNumZ"],
		fSettings["NMI.SynthStepX"],
		fSettings["NMI.SynthStepY"],
		fSettings["NMI.SynthStepZ"],
		fSettings["NMI.WarpStepX"],
		fSettings["NMI.WarpStepY"],
		fSettings["NMI.WarpStepZ"]);
}

void NmiObjects::deleteRating()
{
	int numwarpz = myImage->getNumWarpZ();
	int numwarpy = myImage->getNumWarpY();
	int numwarpx = myImage->getNumWarpX();
	int numsynthz = myRenderer->getNumSynthZ();
	int numsynthy = myRenderer->getNumSynthY();
	int numsynthx = myRenderer->getNumSynthX();

	omp_set_num_threads(nmi_prop_NUMBER_OF_THREADS);
#pragma omp parallel
	{
#pragma omp for schedule(static,1)
		for (int wz = 0; wz < numwarpz; wz++)
		{
			for (int wy = 0; wy < numwarpy; wy++)
			{
				for (int wx = 0; wx < numwarpx; wx++)
				{
					for (int sz = 0; sz < numsynthz; sz++)
					{
						for (int sy = 0; sy < numsynthy; sy++)
						{
							delete rating[wz][wy][wx][sz][sy];
						}
						delete rating[wz][wy][wx][sz];
					}
					delete rating[wz][wy][wx];
				}
				delete rating[wz][wy];
			}
			delete rating[wz];
		}
	}///parallel end
	delete rating;
}

void NmiObjects::resizeKernel(int numsynthx, int numsynthy, int numsynthz, int numwarpx, int numwarpy, int numwarpz, float stepx, float stepy, float stepz, float stepradx, float steprady, float stepradz)
{
	//-----------------------------------------------------DELETE-----------------------------------------------------
	omp_set_num_threads(nmi_prop_NUMBER_OF_THREADS);
#pragma omp parallel
	{
#pragma omp for schedule(static,1)
		for (int wz = 0; wz < myImage->getNumWarpZ(); wz++)
		{
			for (int wy = 0; wy < myImage->getNumWarpY(); wy++)
			{
				for (int wx = 0; wx < myImage->getNumWarpX(); wx++)
				{
					for (int sz = 0; sz < myRenderer->getNumSynthZ(); sz++)
					{
						for (int sy = 0; sy < myRenderer->getNumSynthY(); sy++)
						{
							delete rating[wz][wy][wx][sz][sy];
						}
						delete rating[wz][wy][wx][sz];
					}
					delete rating[wz][wy][wx];
				}
				delete rating[wz][wy];
			}
			delete rating[wz];
		}
	}///parallel end
	delete rating;

	//-----------------------------------------------------CREATE NEW-----------------------------------------------------
	rating = new float*****[numwarpz];
	omp_set_num_threads(nmi_prop_NUMBER_OF_THREADS);
#pragma omp parallel
	{
#pragma omp for schedule(static,1)
		for (int wz = 0; wz < numwarpz; wz++)
		{
			rating[wz] = new float****[numwarpy];
			for (int wy = 0; wy < numwarpy; wy++)
			{
				rating[wz][wy] = new float***[numwarpx];
				for (int wx = 0; wx < numwarpx; wx++)
				{
					rating[wz][wy][wx] = new float**[numsynthz];
					for (int sz = 0; sz < numsynthz; sz++)
					{
						rating[wz][wy][wx][sz] = new float*[numsynthy];
						for (int sy = 0; sy < numsynthy; sy++)
						{
							rating[wz][wy][wx][sz][sy] = new float[numsynthx];
						}
					}
				}
			}
		}
	}///parallel end
}

NmiObjects::~NmiObjects()
{
	std::stringstream ss_log;
	deleteRating();

	ss_log << "Deleting myRenderer" << std::endl;
	delete myRenderer;

	ss_log << "Deleting myImage" << std::endl;
	delete myImage;
	helperFunctions::log(ss_log, logPath);
	delete NmiKernel;
	delete LastNmiKernel;
}

void NmiObjects::setImageVars(int numwarpx, int numwarpy, int numwarpz, float stepradx, float steprady, float stepradz)
{
	myImage->setStepX(stepradx);
	myImage->setStepY(steprady);
	myImage->setStepZ(stepradz);

	myImage->setNumWarpX(numwarpx);
	myImage->setNumWarpY(numwarpy);
	myImage->setNumWarpZ(numwarpz);
}

void NmiObjects::setRendererVars(int numsynthx, int numsynthy, int numsynthz, float stepx, float stepy, float stepz)
{
	myRenderer->setStep_x(stepx);
	myRenderer->setStep_y(stepy);
	myRenderer->setStep_z(stepz);

	myRenderer->setSynthetic_count_x(numsynthx);
	myRenderer->setSynthetic_count_y(numsynthy);
	myRenderer->setSynthetic_count_z(numsynthz);
}

//set MyImage and MyRenderer with concrate values
void  NmiObjects::setNmiObjectsKernel(int numsynthx, int numsynthy, int numsynthz, int numwarpx, int numwarpy, int numwarpz, float stepx, float stepy, float stepz, float stepradx, float steprady, float stepradz)
{
	//deletes the previous version of rating and creates the new version
	resizeKernel(numsynthx, numsynthy, numsynthz, numwarpx, numwarpy, numwarpz, stepx, stepy, stepz, stepradx, steprady, stepradz);

	//deletes the previous version of the images and warping matricies and generates the new structures
	myImage->resizeKernel(numwarpx, numwarpy, numwarpz, stepradx, steprady, stepradz);
	
	//sets the variables in the myRenderer object
	myRenderer->resizeKernel(numsynthx, numsynthy, numsynthz, stepx, stepy, stepz);
}

//set nmiSearchKernel and MyImage and MyRenderer with concrate values from an NmiSearchKernel
void  NmiObjects::setNmiObjectsKernel(NmiSearchKernel * NmiKernel)
{
	setNmiObjectsKernel(NmiKernel->numSynthX, NmiKernel->numSynthY, NmiKernel->numSynthZ, NmiKernel->numWarpX, NmiKernel->numWarpY, NmiKernel->numWarpZ,
		NmiKernel->stepX, NmiKernel->stepY, NmiKernel->stepZ, NmiKernel->stepRadX, NmiKernel->stepRadY, NmiKernel->stepRadZ);
}

//saves NmiKernel to LastNmiKernel, than calculates the new kernel, and sets the NmiObjects
void NmiObjects::NMIobjectsReInitialization() 
{
	//saves the current kernel variables to lastKernel
	LastNmiKernel->setTo(NmiKernel);

	//calculates the new kernel sizes (modifies the nmiSearchKernel
	NmiKernel->resizeKernel();

	//resizes the NMIObjects
	setNmiObjectsKernel(NmiKernel);		
}

void NmiObjects::incN() 
{
	N++;
}

int NmiObjects::getN() 
{
	return N;
}