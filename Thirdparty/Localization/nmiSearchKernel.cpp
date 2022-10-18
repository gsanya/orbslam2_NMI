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
#include "NmiSearchKernel.hpp"
#include "allProperties.hpp"
#include <iomanip>

NmiSearchKernel::NmiSearchKernel( int numsynthx, int numsynthy, int numsynthz, int numwarpx,
	int numwarpy, int numwarpz, float stepx, float stepy, float stepz, float stepradx,
	float steprady, float stepradz):numSynthX(numsynthx), numSynthY(numsynthy), numSynthZ(numsynthz),
	numWarpX(numwarpx), numWarpY(numwarpy),	numWarpZ(numwarpz), stepX(stepx), stepY(stepy),
	stepZ(stepz), stepRadX(stepradx), stepRadY(steprady), stepRadZ(stepradz), bestSynthX(-1), 
	bestSynthY(-1), bestSynthZ(-1), bestWarpX(-1), bestWarpY(-1) , bestWarpZ(-1), NMI(0) {}

NmiSearchKernel::NmiSearchKernel() : numSynthX(-1), numSynthY(-1), numSynthZ(-1),
	numWarpX(-1), numWarpY(-1), numWarpZ(-1), stepX(-1), stepY(-1),
	stepZ(-1), stepRadX(-1), stepRadY(-1), stepRadZ(-1), bestSynthX(-1),
	bestSynthY(-1), bestSynthZ(-1), bestWarpX(-1), bestWarpY(-1), bestWarpZ(-1), NMI(0) {}

void NmiSearchKernel::setKernel(int numsynthx, int numsynthy, int numsynthz, int numwarpx, int numwarpy,
	int numwarpz, float stepx, float stepy, float stepz, float stepradx,
	float steprady, float stepradz)
{
	this->numSynthX=numsynthx;
	this->numSynthY=numsynthy;
	this->numSynthZ=numsynthz;
	this->numWarpX=numwarpx;
	this->numWarpY=numwarpy;
	this->numWarpZ=numwarpz;
	this->stepX=stepx;
	this->stepY=stepy;
	this->stepZ=stepz;
	this->stepRadX=stepradx;
	this->stepRadY=steprady; 
	this->stepRadZ=stepradz;
}

void NmiSearchKernel::setKernel(NmiSearchKernel* NmiKernel)
{
	this->numSynthX = NmiKernel->numSynthX;
	this->numSynthY = NmiKernel->numSynthY;
	this->numSynthZ = NmiKernel->numSynthZ;
	this->numWarpX = NmiKernel->numWarpX;
	this->numWarpY = NmiKernel->numWarpY;
	this->numWarpZ = NmiKernel->numWarpZ;
	this->stepX = NmiKernel->stepX;
	this->stepY = NmiKernel->stepY;
	this->stepZ = NmiKernel->stepZ;
	this->stepRadX = NmiKernel->stepRadX;
	this->stepRadY = NmiKernel->stepRadY;
	this->stepRadZ = NmiKernel->stepRadZ;
}

void NmiSearchKernel::setBest(int  bestsynthx, int bestsynthy, int bestsynthz, int bestwarpx, int bestwarpy, int bestwarpz, float NMI)
{
	this->bestSynthX= bestsynthx;
	this->bestSynthY=bestsynthy;
	this->bestSynthZ= bestsynthz;
	this->bestWarpX= bestwarpx;
	this->bestWarpY = bestwarpy;
	this->bestWarpZ= bestwarpz;
	this->NMI = NMI;
}

void NmiSearchKernel::setBest(NmiSearchKernel* NmiKernel)
{
	this->bestSynthX = NmiKernel->bestSynthX;
	this->bestSynthY = NmiKernel->bestSynthY;
	this->bestSynthZ = NmiKernel->bestSynthZ;
	this->bestWarpX = NmiKernel->bestWarpX;
	this->bestWarpY = NmiKernel->bestWarpY;
	this->bestWarpZ = NmiKernel->bestWarpZ;
}

void NmiSearchKernel::setTo(NmiSearchKernel* NmiKernel) 
{
	setKernel(NmiKernel);
	setBest(NmiKernel);
	NMI = NmiKernel->NMI;
}

bool NmiSearchKernel::isMiddle()
{
	return (bestSynthX == numSynthX/2 && bestSynthY == numSynthY/2 && bestSynthZ == numSynthZ/2 && bestWarpX == numWarpX/2 && bestWarpY == numWarpY /2 && bestWarpZ == numWarpZ/2);
}

void NmiSearchKernel::resizeKernel() 
{
	//if its not on periphery, we reduce the step size by a factor
	if (!((bestSynthX == numSynthX - 1 || bestSynthX == 0) && numSynthX > 1))
		stepX *= nmi_prop_STEPFACTOR;
	if (!((bestSynthY == numSynthY - 1 || bestSynthY == 0) && numSynthY > 1))
		stepY *= nmi_prop_STEPFACTOR;
	if (!((bestSynthZ == numSynthZ - 1 || bestSynthZ == 0) && numSynthZ > 1))
		stepZ *= nmi_prop_STEPFACTOR;
	if (!((bestWarpX == numWarpX - 1 || bestWarpX == 0) && numWarpX > 1))
		stepRadX *= nmi_prop_STEPFACTOR;
	if (!((bestWarpY == numWarpY - 1 || bestWarpY == 0) && numWarpY > 1))
		stepRadY *= nmi_prop_STEPFACTOR;
	if (!((bestWarpZ == numWarpZ - 1 || bestWarpZ == 0) && numWarpZ > 1))
		stepRadZ *= nmi_prop_STEPFACTOR;

	//if along any freedoms the new stepsize is under the min step size, we set the number of steps to 1

	if (stepX < nmi_prop_MIN_KERNEL_TRANSLATION)
		numSynthX = 1;

	if (stepY < nmi_prop_MIN_KERNEL_TRANSLATION)
		numSynthY = 1;

	if (stepZ < nmi_prop_MIN_KERNEL_TRANSLATION)
		numSynthZ = 1;

	if (stepRadX < nmi_prop_MIN_KERNEL_ROTATION)
		numWarpX = 1;

	if (stepRadY < nmi_prop_MIN_KERNEL_ROTATION)
		numWarpY = 1;	

	if (stepRadZ < nmi_prop_MIN_KERNEL_ROTATION)
		numWarpZ = 1;


}

void NmiSearchKernel::resetKernel() 
{
	setKernel(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
}

void NmiSearchKernel::resetBest() 
{
	setBest(-1,-1,-1,-1,-1,-1,0);
}

void NmiSearchKernel::reset()
{
	resetKernel();
	resetBest();
	NMI = 0;
}

int NmiSearchKernel::getNumSynthX() { return numSynthX; }
int NmiSearchKernel::getNumSynthY() { return numSynthY; }
int NmiSearchKernel::getNumSynthZ() { return numSynthZ; }
int NmiSearchKernel::getNumWarpX() { return numWarpX; }
int NmiSearchKernel::getNumWarpY() { return numWarpY; }
int NmiSearchKernel::getNumWarpZ() { return numWarpZ; }

float NmiSearchKernel::getStepX() { return stepX; }
float NmiSearchKernel::getStepY() { return stepY; }
float NmiSearchKernel::getStepZ() { return stepZ; }
float NmiSearchKernel::getStepRadX() { return stepRadX; }
float NmiSearchKernel::getStepRadY() { return stepRadY; }
float NmiSearchKernel::getStepRadZ() { return stepRadZ; }

int NmiSearchKernel::getBestSynthX() { return bestSynthX; }
int NmiSearchKernel::getBestSynthY() { return bestSynthY; }
int NmiSearchKernel::getBestSynthZ() { return bestSynthZ; }
int NmiSearchKernel::getBestWarpX() { return bestWarpX; }
int NmiSearchKernel::getBestWarpY() { return bestWarpY; }
int NmiSearchKernel::getBestWarpZ() { return bestWarpZ; }

float NmiSearchKernel::getNmi() { return NMI; }

std::ostream & operator <<(std::ostream &os, const NmiSearchKernel & NmiKernel)
{
	os.precision(5);
	os << std::fixed;
	os	<< "sX: " << std::setw(2) << NmiKernel.bestSynthX << "/" << std::setw(1) << NmiKernel.numSynthX << ": " << std::setw(6) << NmiKernel.stepX 
		<< ";\t sY: " << std::setw(2) << NmiKernel.bestSynthY << "/" << std::setw(1) << NmiKernel.numSynthY << ": " << std::setw(6) << NmiKernel.stepY 
		<< ";\t sZ: " << std::setw(2) << NmiKernel.bestSynthZ << "/" << std::setw(1) << NmiKernel.numSynthZ << ": " << std::setw(6) << NmiKernel.stepZ 
		<< ";\t rX: " << std::setw(2) << NmiKernel.bestWarpX << "/" << std::setw(1) << NmiKernel.numWarpX << ": " << std::setw(6) << NmiKernel.stepRadX 
		<< ";\t rY: " << std::setw(2) << NmiKernel.bestWarpY << "/" << std::setw(1) << NmiKernel.numWarpY << ": " << std::setw(6) << NmiKernel.stepRadY 
		<< ";\t rZ: " << std::setw(2) << NmiKernel.bestWarpZ << "/" << std::setw(1) << NmiKernel.numWarpZ << ": " << std::setw(6) << NmiKernel.stepRadZ 
		<< ";\t NMI: " << NmiKernel.NMI;
	return os;
}