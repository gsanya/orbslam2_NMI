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

#include <string>

#include "allProperties.hpp"
#include "cameraSettings.hpp"
#include "nmiSearchKernel.hpp"
#include "rendering.hpp"
#include "image.hpp"


class NmiObjects {

public:
	Image * myImage;
	Rendering<nmi_prop_RENDER> *myRenderer;
	float ******rating;//6DoF
	std::string resultsPath;
	std::string logPath;

	//these objects hold the sizes of the Kernel, and the last Kernel(the kernel before the last kernel size change) and some helper functions
	NmiSearchKernel *NmiKernel;
	NmiSearchKernel *LastNmiKernel;
	NmiSearchKernel *InitialNmiKernel;

	NmiObjects(const std::string &strSettingsFile);
	
	~NmiObjects();

	//This is for using
	void setNmiObjectsKernel(int numsynthx, int numsynthy, int numsynthz,  int numwarpx, int numwarpy, int numwarpz, float stepx, float stepy, float stepz, float stepradx, float steprady, float stepradz);
	
	//simple overload to save some typeing
	void setNmiObjectsKernel(NmiSearchKernel * NmiKernel);

	//overload which uses the sizes from the NmiSearchKernel object. Checks if the best match is in any direction at the periphery of the kernel. If it is, it reinitializes the searchkernel and the searchvariables.
	void NMIobjectsReInitialization();

	//Do not use them alone, because they do not resize the rating
	void setRendererVars(int numsyntx, int numsynty, int numsyntz, float stepx, float stepy, float stepz);
	void setImageVars(int numwarpx, int numwarpy, int numwarpz, float stepradx, float steprady, float stepradz);
	
	//increments the N value
	void incN();
	int getN();
private:
	//N is the index of the picture we are currently working on
	int N;
	
	//It deletes the ratings 
	void deleteRating();
	void resizeKernel(int numsynthx, int numsynthy, int numsynthz, int numwarpx, int numwarpy, int numwarpz, float stepx, float stepy, float stepz, float stepradx, float steprady, float stepradz);
};
