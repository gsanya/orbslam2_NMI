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
#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <fstream>
#include <vector>
#include <string>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
// Include GLM
#include <glm/glm.hpp>

#include "ioData.hpp"
#include "cameraSettings.hpp"
#include "allProperties.hpp"

using namespace std;



////////////////////////////////////////////////////////////////////////////////////////
////																				////
////		Képek kimentésére szolgáló file. Itt vannak azok, amik egymásra			////
////		helyezik az image-t és a syntheticet. Ezen kívül itt vannak azok a		////
////		függvények, amik kimentik a dupla képet, ahol az algoritmus eredményt	////
////		és a ground truth-t lehet összehasonlítani.								////
////																				////
////////////////////////////////////////////////////////////////////////////////////////


//ez tölti be a camera settings file-ból a camerasettings vectorba az adatokat
void setCoordinates(const char* fileName, vector<CameraSettings> &cameraSettings)
{
	ifstream infile;
	infile.open(fileName);

	string imageFile;
	cv::Mat K_Mat = cv::Mat::zeros(3, 3, CV_64F);//Matrix to store values;
	cv::Mat Rot = cv::Mat::zeros(3, 3, CV_64F);//Matrix to store values;
	glm::vec3 Pos;
	glm::vec3 Dir;
	glm::vec3 Up;
	string line;

	cameraSettings.clear();



	if (infile.is_open())
	{
		getline(infile, line); //fileName imageWidth, imageHeight
		getline(infile, line); //camera matrix K [3x3]
		getline(infile, line); //radial distortion[3x1]
		getline(infile, line); //tangential distortion[2x1]
		getline(infile, line); //camera position t[3x1]
		getline(infile, line); //camera rotation R[3x3]
		getline(infile, line); //camera model m = K [R|-Rt] X
		do
		{
			imageFile = "";
			infile >> imageFile;
			if (imageFile == "") break;
			getline(infile, line); //imageWidth, imageHeight
			for (int row = 0; row < 3; row++)//K_mat betöltése
			{
				for (int col = 0; col < 3; col++)
				{
					infile >> K_Mat.at<double>(row, col);
				}
				getline(infile, line);
			}
			getline(infile, line); //radial distorsion
			getline(infile, line); //tangential distorsion
			for (int coord = 0; coord < 3; coord++)//camera pos
			{
				infile >> Pos[coord];
			}
			getline(infile, line);
			for (int row = 0; row < 3; row++)//camera rot
			{
				for (int col = 0; col < 3; col++)
				{
					infile >> Rot.at<double>(row, col);
				}
				getline(infile, line);
			}

			Dir = Pos;
			//cv::Mat.at<type>(row,col)
			Dir.x += Rot.at<double>(2, 0);
			Dir.y += Rot.at<double>(2, 1);
			Dir.z += Rot.at<double>(2, 2);

			Up.x = Rot.at<double>(1, 0);
			Up.y = Rot.at<double>(1, 1);
			Up.z = Rot.at<double>(1, 2);



			cameraSettings.push_back(CameraSettings(imageFile, K_Mat, Pos, Dir, Up));
		} while (1);
		infile.close();
	}
	else cout << "Unable to open " << fileName << " file";
};

CameraSettings setupCam(cv::Mat &Pos, cv::Mat &Rot, cv::Mat &K_Mat, cv::Mat &transform, cv::Mat &onlyrot) 
{	
	cv::Mat up_hom;
	cv::Mat dir_hom;
	cv::Mat pos_hom;

	pos_hom = cv::Mat(4, 1, CV_32F);
	up_hom = cv::Mat(4, 1, CV_32F);

	pos_hom.at<float>(0) = Pos.at<float>(0);
	pos_hom.at<float>(1) = Pos.at<float>(1);
	pos_hom.at<float>(2) = Pos.at<float>(2);
	pos_hom.at<float>(3) = 1;

	dir_hom = pos_hom.clone();
	dir_hom.at<float>(0) += Rot.at<float>(2, 0);
	dir_hom.at<float>(1) += Rot.at<float>(2, 1);
	dir_hom.at<float>(2) += Rot.at<float>(2, 2);
	
	up_hom.at<float>(0) = Rot.at<float>(1, 0);
	up_hom.at<float>(1) = Rot.at<float>(1, 1);
	up_hom.at<float>(2) = Rot.at<float>(1, 2);
	up_hom.at<float>(3) = 1;

	pos_hom = transform*pos_hom;
	dir_hom = transform*dir_hom;
	up_hom = onlyrot*up_hom;

	glm::vec3 dir;
	glm::vec3 up;
	glm::vec3 pos;
	
	pos.x = pos_hom.at<float>(0);
	pos.y = pos_hom.at<float>(1);
	pos.z = pos_hom.at<float>(2);

	dir.x = dir_hom.at<float>(0);
	dir.y = dir_hom.at<float>(1);
	dir.z = dir_hom.at<float>(2);

	up.x = up_hom.at<float>(0);
	up.y = up_hom.at<float>(1);
	up.z = up_hom.at<float>(2);


	CameraSettings *settings=new CameraSettings("don't care, doesn't use it", K_Mat, pos, dir, up);
	return *settings;
}

CameraSettings setupCam(cv::Mat &PosInverse, cv::Mat &K_Mat)
{
	glm::vec3 dir;
	glm::vec3 up;
	glm::vec3 pos;
	
	pos.x = PosInverse.at<float>(0,3);
	pos.y = PosInverse.at<float>(1,3);
	pos.z = PosInverse.at<float>(2,3);
	
	dir.x = PosInverse.at<float>(0, 2)+pos.x;
	dir.y = PosInverse.at<float>(1, 2)+pos.y;
	dir.z = PosInverse.at<float>(2, 2)+pos.z;
	
	up.x = PosInverse.at<float>(0, 1);
	up.y = PosInverse.at<float>(1, 1);
	up.z = PosInverse.at<float>(2, 1);
	
	CameraSettings *settings = new CameraSettings("don't care, doesn't uses it", K_Mat, pos, dir, up);
	return *settings;
}

bool saveBMP(const char* fileName, cv::Mat& Image, std::vector<unsigned char>& synthetic)
{
	if (synthetic.size() != Image.cols*Image.rows)
	{
		cout << "The images have not the same size, cannot be made the layout!" << endl;
		return false;
	}
	FILE *f;
	int filesize = 54 + 3 * Image.rows*Image.cols;  //w=Image.cols is your image width, h is image height, both int

	unsigned char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	unsigned char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };
	unsigned char bmppad[3] = { 0,0,0 };

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(Image.cols);
	bmpinfoheader[5] = (unsigned char)(Image.cols >> 8);
	bmpinfoheader[6] = (unsigned char)(Image.cols >> 16);
	bmpinfoheader[7] = (unsigned char)(Image.cols >> 24);
	bmpinfoheader[8] = (unsigned char)(Image.rows);
	bmpinfoheader[9] = (unsigned char)(Image.rows >> 8);
	bmpinfoheader[10] = (unsigned char)(Image.rows >> 16);
	bmpinfoheader[11] = (unsigned char)(Image.rows >> 24);

	f = fopen(fileName, "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	unsigned char* img = Image.data;
	unsigned char* syn = (unsigned char*)&synthetic[0];
	unsigned char null = 0;
	for (int i = 0; i<Image.rows; i++)
	{
		for (int j = 0; j < Image.cols; j++)
		{
			fwrite(img + (Image.cols*(i)) + j, 1, 1, f);
			fwrite(syn + (Image.cols*(i)) + j, 1, 1, f);
			fwrite(&null, 1, 1, f);
		}
		fwrite(bmppad, 1, (4 - (Image.cols * 3) % 4) % 4, f);
	}
	fclose(f);

}

bool saveImage(const char* fileName, const int& height, const int& width, std::vector<unsigned char>& synthetic)
{
	cv::Mat color = cv::Mat::zeros(height, width, CV_8U);
	cv::cvtColor(color, color, CV_GRAY2BGR);
	for (int i = 0; i < color.cols; i++)
	{
		for (int j = 0; j < color.rows; j++)
		{
			color.at<cv::Vec3b>(color.rows - 1 - j, i).val[0] = synthetic[j*color.cols + i];
			color.at<cv::Vec3b>(color.rows - 1 - j, i).val[1] = synthetic[j*color.cols + i];
			color.at<cv::Vec3b>(color.rows - 1 - j, i).val[2] = synthetic[j*color.cols + i];
		}
	}
	//cv::resize(color, color, cv::Size(), 1.0f / 2.0f, 1.0f / 2.0f, cv::INTER_LANCZOS4);
	imwrite(fileName, color);
	return true;
}

//összességében õ kap egy warpolt képet, és egy synthetic képet (az elõbbi cv::Mat, míg utóbbi egy uchar vektor) És ezeket a képeket "fekteti egymásra" úgy, h a kéket nullázza, a piros lesz a warpolt, a zöld pedig a synthetic
cv::Mat getImage(cv::Mat& Image, std::vector<unsigned char>& synthetic)
{
	cv::Mat color;
	//ez valahogy az Image-bõl színest csinál a color-ba (csak formailag) tehát most már a tárolóba elfér a színinfó
	cv::cvtColor(Image, color, cv::COLOR_GRAY2BGR);
	//végig megyünk a color mátrixon

	for (int i = 0; i < color.cols; i++)
	{
		for (int j = 0; j < color.rows; j++)
		{
			//kék minden pontban 0
			color.at<cv::Vec3b>(j, i).val[0] = 0;
			//pirosat hagyja
			//zöld minden pontban annyi, mint a synthetic
			color.at<cv::Vec3b>(color.rows - 1 - j, i).val[1] = synthetic[j*color.cols + i]; //a synthetic agy sima vektor, szóval ahoz, h a jedik sor iedik elemét érjük el, így kell megfogalmazni; a másik meg alulról megy felfelé
		}
	}

	//felezzük mindkét oldalát //nemtom miért ilyen furán írta a 0.5-öt //utsó paraméter az interpoláció típusa (enum)
	cv::resize(color, color, cv::Size(), 1.0f / 2.0f, 1.0f / 2.0f, cv::INTER_LANCZOS4);
	return color;
}

cv::Mat getImageNoHalf(cv::Mat& Image, std::vector<unsigned char>& synthetic)
{
	cv::Mat color;
	//ez valahogy az Image-bõl színest csinál a color-ba (csak formailag) tehát most már a tárolóba elfér a színinfó
	cv::cvtColor(Image, color, cv::COLOR_GRAY2BGR);
	//végig megyünk a color mátrixon

	for (int i = 0; i < color.cols; i++)
	{
		for (int j = 0; j < color.rows; j++)
		{
			//kék minden pontban 0
			color.at<cv::Vec3b>(j, i).val[0] = 0;
			//pirosat hagyja
			//zöld minden pontban annyi, mint a synthetic
			color.at<cv::Vec3b>(color.rows - 1 - j, i).val[1] = synthetic[j*color.cols + i]; //a synthetic agy sima vektor, szóval ahoz, h a jedik sor iedik elemét érjük el, így kell megfogalmazni; a másik meg alulról megy felfelé
		}
	}

	//felezzük mindkét oldalát //nemtom miért ilyen furán írta a 0.5-öt //utsó paraméter az interpoláció típusa (enum)
	//cv::resize(color, color, cv::Size(), 1.0f / 2.0f, 1.0f / 2.0f, cv::INTER_LANCZOS4);
	return color;
}

//összességében õ kap bemenetnek 2 paramétersorhoz(synx,syny,ynz,warpz) warped és a synthetic képeket, össze piroszöldeli, és egymás mellé teszi
bool saveImage(const char* fileName,
	cv::Mat& Image1,
	std::vector<unsigned char>& synthetic1,
	cv::Mat& Image2,
	std::vector<unsigned char>& synthetic2)
{
	using namespace cv;
	//a 2 bemenõ mátrixból csinál két zöld-piros képet a fentebbi függvénnyel
	Mat color1 = getImage(Image1, synthetic1);
	Mat color2 = getImage(Image2, synthetic2);
	Size sz1 = color1.size();
	Size sz2 = color2.size();
	//a két elkészült képet egmás mellé teszi. balra megy a color 1
	Mat result(sz1.height, sz1.width + sz2.width, CV_8UC3);
	Mat left(result, Rect(0, 0, sz1.width, sz1.height));
	color1.copyTo(left);
	//jobbra megy a color kettõ
	Mat right(result, Rect(sz1.width, 0, sz2.width, sz2.height));
	color2.copyTo(right);
	cv::imwrite(fileName, result);
	return true;
}


//for saving the image from the new position
bool saveImage(const char* fileName,
	cv::Mat& Image,
	std::vector<unsigned char>& synthetic)
{
	cv::Mat colored = getImageNoHalf(Image, synthetic);
	imwrite(fileName, colored);
	return true;
}