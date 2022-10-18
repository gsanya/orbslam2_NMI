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

#include<iostream>


class NmiSearchKernel {
public:
	//basic variables about kernel size and resolution
	//they are public, so they are easier to reach
	int  numSynthX, numSynthY, numSynthZ, numWarpX, numWarpY, numWarpZ;
	float stepX, stepY, stepZ, stepRadX, stepRadY, stepRadZ;

	float NMI;

	//variables about the best match
	int  bestSynthX, bestSynthY, bestSynthZ, bestWarpX, bestWarpY, bestWarpZ;

public:
	NmiSearchKernel(int numsynthx, int numsynthy, int numsynthz, int numwarpx, int numwarpy, int numwarpz, float stepx, float stepy, float stepz, float stepradx, float steprady, float stepradz);
	NmiSearchKernel();

	void setKernel(int numsynthx, int numsynthy, int numsynthz, int numwarpx, int numwarpy, int numwarpz, float stepx, float stepy, float stepz, float stepradx, float steprady, float stepradz);
	void setKernel(NmiSearchKernel* NmiKernel);
	void setBest(int  bestsynthx, int bestsynthy, int bestsynthz, int bestwarpx, int bestwarpy, int bestwarpz, float NMI);
	void setBest(NmiSearchKernel* NmiKernel);
	void setTo(NmiSearchKernel* NmiKernel);

	//returns true if the best match is in the middle of the kernel
	bool isMiddle();

	//it resizes the kernel with 2 steps in each directions where the best match is on the periphery
	//it only resizes this class (we should call NMIobjectsReInitialization after that, so that it changes the real kernel sizes)
	void resizeKernel();

	//reset functions
	void resetKernel();
	void resetBest();
	void reset();

	//operator redefinition
	friend std::ostream & operator <<(std::ostream &os, const NmiSearchKernel & NmiKernel);

	//simple get functions:
	int getNumSynthX();
	int getNumSynthY();
	int getNumSynthZ();
	int getNumWarpX();
	int getNumWarpY();
	int getNumWarpZ();

	float getStepX();
	float getStepY();
	float getStepZ();
	float getStepRadX();
	float getStepRadY();
	float getStepRadZ();

	int getBestSynthX();
	int getBestSynthY();
	int getBestSynthZ();
	int getBestWarpX();
	int getBestWarpY();
	int getBestWarpZ();

	float getNmi();

};
std::ostream & operator <<(std::ostream &os, const NmiSearchKernel & NmiKernel);