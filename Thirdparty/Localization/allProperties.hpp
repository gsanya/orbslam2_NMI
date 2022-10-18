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
				////////////////////////////////////////////////////////////////////////////////////
				////																			////
				////				--SOME PROPERTIES THAT CAN BE CHANGED--						////
				////																			////
				////////////////////////////////////////////////////////////////////////////////////

#define nmi_prop_MAX_ITERATION_COUNT 4
//log some additional information and saves nice pictures
#define orb_prop_log false
//the algorithm performs relocalization at every 'nmi_prop_RELOC_FREQUENCY'th keyframe
#define nmi_prop_RELOC_FREQUENCY 2
//when the best match is in the middle in a direction (out of 6), the stepzises in that direction are multiplied by nmi_prop_STEPFACTOR
#define nmi_prop_STEPFACTOR 0.5f
//number of threads for the cpu process (only some creation a deletion is parallelized)
#define nmi_prop_NUMBER_OF_THREADS 12
//test property. If set to true, the background pixels (0 or 255) are also used to calculate the NMI. If set to false, those pixels are ignored. No real difference
//was observed
#define nmi_prop_BG true
//defines the type of renderer to use (1 is for obj mesh with bmp texture, 4 is for xyz pointcloud with RGB information)
//other rendering options can be added
#define nmi_prop_RENDER 1 //4 if point cloud 1 if mesh
//shaders are used by OpenGL for rendering. This specifies the location of those shaders.
#define nmi_prop_SHADER_LOC "../Thirdparty/Localization/shaders/"
//the output will be saved here under a subdirectory with the date and time. ORBSLAM_NMI_results folder should be created, or the program will return with error.
//can be absolute, or relative directory (if Visual studio runs the code with local debugger, the working directory is where the project files are, and not the location of the exe)
#define nmi_prop_OUTPUT_LOC "/ORBSLAM_NMI_results"
//these parameters are the thresholds for the step numbers. If the calculated step size in one iteration is under this threshold, there will be
//no pose refinement in that direction (different threshold for rotation and translation)
#define nmi_prop_MIN_KERNEL_ROTATION 0.001 //rad
#define nmi_prop_MIN_KERNEL_TRANSLATION 0.005 //m