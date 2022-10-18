/**
* This file is part of orbslam2_NMI.
*
* Copyright (C) 2021 Sándor Gazdag <gazdag.sandor at sztaki dot hu> (SZTAKI)
* For more information see <https://github.com/gsanya/orbslam2_NMI>
*
* ORB-orbslam2_NMI is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"

#include"localization.hpp"
#include"rendering.hpp"
#include"allProperties.hpp"
#include"image.hpp"
#include"helperFunctions.hpp"


using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
	vector<double> &vTimestamps);


int main(int argc, char **argv)
{
	std::stringstream ss_log;
	
	if (argc != 4)
	{
		cerr << endl << "Usage: ./mono_newer_college path_to_vocabulary path_to_settings path_to_sequence" << endl;
		return 1;
	}

	

	// Retrieve paths to images
	vector<string> vstrImageFilenames;
	vector<double> vTimestamps;
	LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);
	int nImages = vstrImageFilenames.size();

	if (nImages <= 0)
	{
		cerr << "ERROR: Failed to load images" << endl;
		return 1;
	}
	//here I initialize the objects required for NMI relocalization
	NmiObjects*myObjects = new NmiObjects(argv[2]);
	


	

	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, myObjects, true);
	

	// Vector for tracking time statistics
	vector<float> vTimesTrack;
	vTimesTrack.resize(nImages);

	cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);

	ss_log << "\nNMI relocalization options:\n" << "nmi_prop_RELOC_FREQUENCY: " << nmi_prop_RELOC_FREQUENCY << "\tNMI Initial treshold: " << float(fSettings["NMI.Treshold"]) << "\nnmi_prop_MAX_ITERATION_COUNT: " << nmi_prop_MAX_ITERATION_COUNT;
	if (nmi_prop_RENDER == 4)
		ss_log << "\nPoint size: " << float(fSettings["NMI.Render.PointSize"]) << "\tNear Plane: " << float(fSettings["NMI.Render.NearPlane"]) << "\tFar Plane: " << float(fSettings["NMI.Render.FarPlane"]);
	ss_log << endl << "-------" << endl;
	ss_log << "Start processing sequence ..." << endl;
	ss_log << "Images in the sequence: " << nImages << endl << endl;
	helperFunctions::log(ss_log, myObjects->logPath);
	// Main loop
	cv::Mat im;
	for (int ni = 0; ni< nImages; ni++, myObjects->incN())
	{
		ss_log << "\n/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\\nFRAME BEGIN\n";
		ss_log << myObjects->getN() << "\t filename:\t" << vstrImageFilenames[ni] << "\n";
		helperFunctions::log(ss_log, myObjects->logPath);
		// Read image from file
		im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
		double tframe = vTimestamps[ni];

		if (im.empty())
		{
			cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << ni << endl;
			
			return 1;
		}
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		
		// Pass the image to the SLAM system
		SLAM.TrackMonocular(im, tframe);
			
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

		double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
				
		vTimesTrack[ni] = ttrack;

		// Wait to load the next frame
		double T = 0;
		if (ni<nImages - 1)
			T = vTimestamps[ni + 1] - tframe;
		else if (ni>0)
			T = tframe - vTimestamps[ni - 1];

		if (ttrack<T)
			usleep((T - ttrack)*1e6);

		ss_log << "time: " << ttrack << "\n"<<"FRAME END\n/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\\n";

		helperFunctions::log(ss_log, myObjects->logPath);
		if (ni%100== 0 && ni>1)
			SLAM.SaveFullTrajectory(myObjects->resultsPath + "/FrameTrajectory");
	}
	

	// Stop all threads
	SLAM.Shutdown();

	

	// Tracking time statistics
	sort(vTimesTrack.begin(), vTimesTrack.end());
	float totaltime = 0;
	for (int ni = 0; ni<nImages; ni++)
	{
		totaltime += vTimesTrack[ni];
	}
	ss_log << "-------" << endl << endl;
	ss_log << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
	ss_log << "mean tracking time: " << totaltime / nImages << endl;
	helperFunctions::log(ss_log, myObjects->logPath);
	// Save camera trajectory
	SLAM.SaveFullTrajectory(myObjects->resultsPath + "/FrameTrajectory");
	SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

	std::stringstream path;
	path << SLAM.myObjects->resultsPath << "/FinalMap.xyz";
	SLAM.getTracker()->PrintMapPoints(path.str(), 127, 255, 127);
	delete myObjects;

	return 0;
}

void LoadImages(const std::string &strPathToSequence, std::vector<std::string> &vstrImageFilenames, std::vector<double> &vTimestamps)
{
	//opens, and loads timestamp data into a vector
	std::ifstream fFileNames;
	std::string strPathTimeFile = strPathToSequence + "/_files.txt";

	fFileNames.open(strPathTimeFile.c_str());
	int i = 0;
	int startUnixTimeStamp = 0;
	//int startMicroSecs = 0;
	while (!fFileNames.eof())
	{
		std::string s;
		std::getline(fFileNames, s);
		if (!s.empty())
		{
			vstrImageFilenames.push_back(strPathToSequence + "/" + s);
			//end of x substring
			s = s.erase(s.find('.'), 4);
			s = s.erase(0, 7);
			s = s.erase(s.find('_'), 1);
			s = s.erase(16, 3);
			std::string s1 = s.substr(0, 10);
			std::string s2 = s.substr(10, 6);
			int unixTimeStamp = std::stoi(s1);
			int microSecs = std::stoi(s2);
			if (i == 0)
			{
				startUnixTimeStamp = unixTimeStamp;
				//startMicroSecs = microSecs;
			}


			double t = (unixTimeStamp - startUnixTimeStamp) + microSecs * 0.000001;

			vTimestamps.push_back(t);
			i++;
		}
	}
	//the folder containing the images
}
