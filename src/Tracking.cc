/**
* This file was originally part of ORB-SLAM2. Now it is part of orbslam2_NMI, but most parts
* of it is the original, so we left the original license and author here. For more 
* information about orbslam2_NMI see <https://github.com/gsanya/orbslam2_NMI>
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
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


#include"Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/cudawarping.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"
#include"Optimizer.h"
#include"PnPsolver.h"

#include"kernel.cuh"

#include<iostream>
#include<mutex>
#include<ctime>

#include <boost/filesystem.hpp>

using namespace std;

std::string sStates[] = { "NO_IMAGES_YET","NOT_INITIALIZED", "OK", "LOST" };

namespace ORB_SLAM2
{

cv::Mat toHomogeneous(cv::Mat const &p)
{
	cv::Mat hp(4, 1, CV_32F);
	hp.at<float>(0) = p.at<float>(0);
	hp.at<float>(1) = p.at<float>(1);
	hp.at<float>(2) = p.at<float>(2);
	hp.at<float>(3) = 1;
	return hp;
}

cv::Mat fromHomogeneous(cv::Mat const &p)
{
	cv::Mat hp(3, 1, CV_32F);
	hp.at<float>(0) = p.at<float>(0) / p.at<float>(3);
	hp.at<float>(1) = p.at<float>(1) / p.at<float>(3);
	hp.at<float>(2) = p.at<float>(2) / p.at<float>(3);
	return hp;
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(cv::Mat &R){
	cv::Mat Rt;
	cv::transpose(R, Rt);
	cv::Mat shouldBeIdentity = Rt * R;
	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
	return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R){
		assert(isRotationMatrix(R));

		float sy = sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));
		bool singular = sy < 1e-6; // If
		float x, y, z;
		if (!singular){
				x = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
				y = atan2(-R.at<float>(2, 0), sy);
				z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
		}
		else{
				x = atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
				y = atan2(-R.at<float>(2, 0), sy);
				z = 0;
		}
		return cv::Vec3f(x, y, z);
}


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, NmiObjects *myObjectsPar):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0), keyFramesSinceLastNmi(0)
{
	std::stringstream ss_log;

	mDistanceSinceLastNMI = cv::Vec3f(0.0, 0.0, 0.0);
	mRotationSinceLastNMI = cv::Vec3f(0.0, 0.0, 0.0);

	//setting up pointers for NMI relocalization
	MyObjects = myObjectsPar;	
	MyImage = MyObjects->myImage;
	MyRenderer = MyObjects->myRenderer;
	mbSaveAllImages = false;

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

	cv::Mat Init1 = cv::Mat::zeros(4, 4, CV_32F);
	cv::Mat Init2 = cv::Mat::zeros(4, 4, CV_32F);
	fSettings["NMI.Init1"] >> Init1;
	fSettings["NMI.Init2"] >> Init2;
	mnInitOffset = (int)fSettings["NMI.Offset"];
	mfNmiInitTresholf = fSettings["NMI.Treshold"];
	Init1.copyTo(mInit1);
	Init2.copyTo(mInit2);

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 10;
    mMaxFrames = fps;

	ss_log<< endl << "Camera Parameters: " << endl;
	ss_log<< "- fx: " << fx << endl;
	ss_log << "- fy: " << fy << endl;
	ss_log << "- cx: " << cx << endl;
	ss_log << "- cy: " << cy << endl;
	ss_log << "- k1: " << DistCoef.at<float>(0) << endl;
	ss_log << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
		ss_log << "- k3: " << DistCoef.at<float>(4) << endl;
	ss_log << "- p1: " << DistCoef.at<float>(2) << endl;
	ss_log << "- p2: " << DistCoef.at<float>(3) << endl;
	ss_log << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
		ss_log << "- color order: RGB (ignored if grayscale)" << endl;
    else
		ss_log << "- color order: BGR (ignored if grayscale)" << endl;
	helperFunctions::log(ss_log, MyObjects->logPath);
    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

	ss_log << endl  << "ORB Extractor Parameters: " << endl;
	ss_log << "- Number of Features: " << nFeatures << endl;
	ss_log << "- Scale Levels: " << nLevels << endl;
	ss_log << "- Scale Factor: " << fScaleFactor << endl;
	ss_log << "- Initial Fast Threshold: " << fIniThFAST << endl;
	ss_log << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
		ss_log << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }
	helperFunctions::log(ss_log, MyObjects->logPath);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

//I don't use this
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

	
    Track();

	
    return mCurrentFrame.mTcw.clone();
}

//I don't use this
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

//This is the one I use
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
	std::chrono::steady_clock::time_point t1;
	std::chrono::steady_clock::time_point t2;
	double ttrack;

	mImGray = im;
	//convert image to grayscale
	if (mImGray.channels() == 3)
	{
		if (mbRGB)
		{
			cvtColor(mImGray, mImGray, CV_RGB2GRAY);
		}
		else
		{
			cvtColor(mImGray, mImGray, CV_BGR2GRAY);
		}
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }
	//if it's the first image (InitOrbExtractor??)
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);//sets no position (Tcw)
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
	

	Track();//it will track the position of the frame



    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
	std::stringstream ss_log;
////--------------------------------------------------For BenchMarking--------------------------------------------------////
	ss_log<< "MapPoints size: " << this->mpMap->GetAllMapPoints().size() << "\n";
	helperFunctions::log(ss_log, MyObjects->logPath);
////-----------------------------------------------End of Benchmarking part---------------------------------------------////
	
	std::chrono::steady_clock::time_point t1;
	std::chrono::steady_clock::time_point t2;
	double ttrack;

	if (mState == NO_IMAGES_YET)
	{
		mState = NOT_INITIALIZED;
	}

	mLastProcessedState = mState;

	// Get Map Mutex -> Map cannot be changed
	//it is held till the end ofthe block (end of track() function)
	unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
	

	if (mState == NOT_INITIALIZED)
	{
		if (mSensor == System::STEREO || mSensor == System::RGBD)
			StereoInitialization();
		else
		{
			if(mCurrentFrame.mnId==0|| mCurrentFrame.mnId == mnInitOffset)
				InitializeWithNMI();				
		}
		mpFrameDrawer->Update(this);

		if (mState != OK)
		{		
			return;
		}
	}
	else
	{
		// System is initialized. Track Frame.
		bool bOK;

		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)
		if (!mbOnlyTracking)//SLAM mode
		{
			// Local Mapping is activated. This is the normal behaviour, unless
			// you explicitly activate the "only tracking" mode.

			if (mState == OK)
			{
				// Local Mapping might have changed some MapPoints tracked in last frame, we update it.
				CheckReplacedInLastFrame();

				if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)//right after relocalization
				{
					if (orb_prop_log)
					{
						ss_log << std::endl << "TrackReferenceKeyFrame() Matches:\n";
						helperFunctions::log(ss_log, MyObjects->logPath);
					}
					bOK = TrackReferenceKeyFrame();//this function will only be called once, if no relocalization, and succesfull motionmodell tracking
					mnMatchesInliers = 0;
					for (int i = 0; i < mCurrentFrame.N; i++)
					{
						if ((mCurrentFrame.mvpMapPoints[i]) && (!mCurrentFrame.mvbOutlier[i]) && (mCurrentFrame.mvpMapPoints[i]->Observations() > 0))
							mnMatchesInliers++;
					}

					ss_log << " Inlier Matches after TrackReferenceKeyFrame(): " << mnMatchesInliers << "\n";
					helperFunctions::log(ss_log, MyObjects->logPath);
				}
				else
				{
					bOK = TrackWithMotionModel();//sets the pose with velocity, than optimizes it with optimizer. It also fills the mappoints
					if (!bOK)
						bOK = TrackReferenceKeyFrame();

					mnMatchesInliers = 0;
					for (int i = 0; i < mCurrentFrame.N; i++)
					{
						if ((mCurrentFrame.mvpMapPoints[i]) && (!mCurrentFrame.mvbOutlier[i]) && (mCurrentFrame.mvpMapPoints[i]->Observations() > 0))
							mnMatchesInliers++;
					}

					ss_log <<" Inlier Matches after TrackWithMotionModel(): " << mnMatchesInliers << "\n";
					helperFunctions::log(ss_log, MyObjects->logPath);


				}
			}
			else//if tracking was lost
			{
				bOK = Relocalization();
			}
		}
		else//tracking only mode (I don't use this)
		{
			// Localization Mode: Local Mapping is deactivated

			if (mState == LOST)
			{
				bOK = Relocalization();
			}
			else
			{
				if (!mbVO)
				{
					// In last frame we tracked enough MapPoints in the map

					if (!mVelocity.empty())
					{
						bOK = TrackWithMotionModel();
					}
					else
					{
						bOK = TrackReferenceKeyFrame();
					}
				}
				else
				{
					// In last frame we tracked mainly "visual odometry" points.

					// We compute two camera poses, one from motion model and one doing relocalization.
					// If relocalization is sucessfull we choose that solution, otherwise we retain
					// the "visual odometry" solution.

					bool bOKMM = false;
					bool bOKReloc = false;
					vector<MapPoint*> vpMPsMM;
					vector<bool> vbOutMM;
					cv::Mat TcwMM;
					if (!mVelocity.empty())//If there is velocity
					{
						bOKMM = TrackWithMotionModel();//sets the pose with velocity, than optimizes it with optimizer. It also fills the mappoints
						vpMPsMM = mCurrentFrame.mvpMapPoints;
						vbOutMM = mCurrentFrame.mvbOutlier;
						TcwMM = mCurrentFrame.mTcw.clone();
					}
					bOKReloc = Relocalization();

					if (bOKMM && !bOKReloc)//If Motion Model was succesful, but relocalization wasn't
					{
						mCurrentFrame.SetPose(TcwMM);
						mCurrentFrame.mvpMapPoints = vpMPsMM;
						mCurrentFrame.mvbOutlier = vbOutMM;

						if (mbVO)
						{
							for (int i = 0; i < mCurrentFrame.N; i++)
							{
								if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
								{
									mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
								}
							}
						}
					}
					else if (bOKReloc)//If Motion Model wasn't succesful, or relocaliztion was.
					{
						mbVO = false;
					}

					bOK = bOKReloc || bOKMM;
				}
			}
		}

		//The referencekeyframe is added to the current frame. The names are the same, but on the left it is a part of the frame, and on the right it is the part of tracking, which constantly updates.
		mCurrentFrame.mpReferenceKF = mpReferenceKF;

		// If we have an initial estimation of the camera pose and matching. Track the local map.
		if (!mbOnlyTracking)//full SLAM mode
		{
			if (bOK)
			{
				bOK = TrackLocalMap();
				mnMatchesInliers = 0;
				for (int i = 0; i < mCurrentFrame.N; i++)
				{
					if ((mCurrentFrame.mvpMapPoints[i]) && (!mCurrentFrame.mvbOutlier[i]) && (mCurrentFrame.mvpMapPoints[i]->Observations() > 0))
						mnMatchesInliers++;
				}

				ss_log << "Inlier Matches after TrackLocalMap(): " << mnMatchesInliers << "\n";
				helperFunctions::log(ss_log, MyObjects->logPath);
			}	
		}
		else //I don't use this
		{
			// mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
			// a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
			// the camera we will use the local map again.
			if (bOK && !mbVO)
				bOK = TrackLocalMap();
		}

			if (bOK)//if both the pose estimation and the tracking were ok
			mState = OK;
		else
			mState = LOST;


		// Update drawer (visualization)
		mpFrameDrawer->Update(this);

		// If tracking were good, check if we insert a keyframe
		if (bOK){
			// Update motion model
			if (!mLastFrame.mTcw.empty()){
				cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
				mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
				mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
				mVelocity = mCurrentFrame.mTcw*LastTwc;
			}
			else
				mVelocity = cv::Mat();

			//save framepose here
			mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

			// Clean VO matches
			for (int i = 0; i < mCurrentFrame.N; i++){
				MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
				if (pMP)
					if (pMP->Observations() < 1){
						mCurrentFrame.mvbOutlier[i] = false;
						mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
					}
			}

			// Delete temporal MapPoints
			for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++){
				MapPoint* pMP = *lit;
				delete pMP;
			}
			mlpTemporalPoints.clear();


			// Check if we need to insert a new keyframe
			if (NeedNewKeyFrame()){
				if (keyFramesSinceLastNmi+1 == nmi_prop_RELOC_FREQUENCY){
					RelocalizeWithNMIStrategy((void*)&mCurrentFrame, false);
					if (mCurrentFrame.GetNMIRelocalized()) {
						if (MyObjects->NmiKernel->getNumSynthX() > 1)
							mDistanceSinceLastNMI(0) = 0.0;
						if (MyObjects->NmiKernel->getNumSynthY() > 1)
							mDistanceSinceLastNMI(1) = 0.0;
						if (MyObjects->NmiKernel->getNumSynthZ() > 1)
							mDistanceSinceLastNMI(2) = 0.0;
						if (MyObjects->NmiKernel->getNumWarpX() > 1)
							mRotationSinceLastNMI(0) = 0.0;
						if (MyObjects->NmiKernel->getNumWarpY() > 1)
							mRotationSinceLastNMI(1) = 0.0;
						if (MyObjects->NmiKernel->getNumWarpZ() > 1)
							mRotationSinceLastNMI(2) = 0.0;
					}				
				}
				CreateNewKeyFrame();
				if (keyFramesSinceLastNmi==nmi_prop_RELOC_FREQUENCY)
					keyFramesSinceLastNmi = 0;

				if (mCurrentFrame.GetNMIRelocalized())
					mCurrentFrame.mpReferenceKF->SetNMIRelocalized(true);
				//if at least 20 keyframes were added since last optimization I optimize with
			}

			// We allow points with high innovation (considererd outliers by the Huber Function)
			// pass to the new keyframe, so that bundle adjustment will finally decide
			// if they are outliers or not. We don't want next frame to estimate its position
			// with those points so we discard them in the frame.
			for (int i = 0; i < mCurrentFrame.N; i++){
				if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
			}

		}

		// Reset if the camera get lost soon after initialization
		if (mState == LOST){
			if (mpMap->KeyFramesInMap() <= 5){
				ss_log << "Track lost soon after initialisation, reseting...\n";
				helperFunctions::log(ss_log, MyObjects->logPath);
				mpSystem->Reset();
			}
		}

		if (!mCurrentFrame.mpReferenceKF)
			mCurrentFrame.mpReferenceKF = mpReferenceKF;
		
		if (!mCurrentFrame.mTcw.empty() && !mLastFrame.mTcw.empty()){
			cv::Mat Twc_current = mCurrentFrame.mTcw.clone();
			cv::Mat Twc_last = mLastFrame.mTcw.clone();
			mDistanceSinceLastNMI[0] += abs(Twc_current.at<float>(0, 3) - Twc_last.at<float>(0, 3));
			mDistanceSinceLastNMI[1] += abs(Twc_current.at<float>(1, 3) - Twc_last.at<float>(1, 3));
			mDistanceSinceLastNMI[2] += abs(Twc_current.at<float>(2, 3) - Twc_last.at<float>(2, 3));

			cv::Rect r(0, 0, 3, 3);
			cv::Vec3f rotdiff = rotationMatrixToEulerAngles(Twc_current(r))- rotationMatrixToEulerAngles(Twc_last(r));
			rotdiff(0) = abs(rotdiff(0));
			rotdiff(1) = abs(rotdiff(1));
			rotdiff(2) = abs(rotdiff(2));
			
			mRotationSinceLastNMI += rotdiff;
		}
		mLastFrame = Frame(mCurrentFrame);		
	}
	
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
	if (!mCurrentFrame.mTcw.empty())
	{
		cv::Mat Tcr;
		//some small mathematical errors are always present
		float x = cv::sum(abs(mCurrentFrame.mTcw - mCurrentFrame.mpReferenceKF->GetPose()))[0];
		if ( x< 0.01)
			Tcr = cv::Mat::eye(4, 4, CV_32F);
		else
			Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();

        mlRelativeFramePoses.push_back(Tcr);
		mlFrameNumberList.push_back(mCurrentFrame.mnId);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
		
    }
    else{
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
		mlFrameNumberList.push_back(mCurrentFrame.mnId);
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}

//I don't use this
void Tracking::StereoInitialization()
{
	std::stringstream ss_log;

	if (mCurrentFrame.N > 500)
	{
		// Set Frame pose to the origin
		mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

		// Create KeyFrame
		KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

		// Insert KeyFrame in the map
		mpMap->AddKeyFrame(pKFini);

		// Create MapPoints and asscoiate to KeyFrame
		for (int i = 0; i < mCurrentFrame.N; i++)
		{
			float z = mCurrentFrame.mvDepth[i];
			if (z > 0)
			{
				cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
				MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
				pNewMP->AddObservation(pKFini, i);
				pKFini->AddMapPoint(pNewMP, i);
				pNewMP->ComputeDistinctiveDescriptors();
				pNewMP->UpdateNormalAndDepth();
				mpMap->AddMapPoint(pNewMP);


	 
                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        ss_log << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;
		helperFunctions::log(ss_log, MyObjects->logPath);

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

//I don't use this
void Tracking::MonocularInitialization()
{

	if (!mpInitializer)
	{
		// Set Reference Frame
		if (mCurrentFrame.mvKeys.size() > 100)
		{
			mInitialFrame = Frame(mCurrentFrame);
			mLastFrame = Frame(mCurrentFrame);
			mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
			for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
				mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

			if (mpInitializer)
				delete mpInitializer;

			mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

			fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

			return;
		}
	}
	else
	{
		// Try to initialize
		if ((int)mCurrentFrame.mvKeys.size() <= 100)
		{
			delete mpInitializer;
			mpInitializer = static_cast<Initializer*>(NULL);
			fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
			return;
		}

		// Find correspondences
		ORBmatcher matcher(0.9, true);
		int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

		// Check if there are enough correspondences
		if (nmatches < 100)
		{
			delete mpInitializer;
			mpInitializer = static_cast<Initializer*>(NULL);
			return;
		}

		cv::Mat Rcw; // Current Camera Rotation
		cv::Mat tcw; // Current Camera Translation
		vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

		if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
		{
			for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
			{
				if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
				{
					mvIniMatches[i] = -1;
					nmatches--;
				}
			}

			// Set Frame Poses
			mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
			cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
			Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
			tcw.copyTo(Tcw.rowRange(0, 3).col(3));
			mCurrentFrame.SetPose(Tcw);

			CreateInitialMapMonocular();
		}
	}
}

//I don't use this
void Tracking::CreateInitialMapMonocular()
{
	// Create KeyFrames
	KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
	KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);


	pKFini->ComputeBoW();
	pKFcur->ComputeBoW();

	// Insert KFs in the map
	mpMap->AddKeyFrame(pKFini);
	mpMap->AddKeyFrame(pKFcur);

	// Create MapPoints and asscoiate to keyframes
	for (size_t i = 0; i < mvIniMatches.size(); i++)
	{
		if (mvIniMatches[i] < 0)
			continue;

		//Create MapPoint.
		cv::Mat worldPos(mvIniP3D[i]);

		MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

		pKFini->AddMapPoint(pMP, i);
		pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

		pMP->AddObservation(pKFini, i);
		pMP->AddObservation(pKFcur, mvIniMatches[i]);

		pMP->ComputeDistinctiveDescriptors();
		pMP->UpdateNormalAndDepth();

		//Fill Current Frame structure
		mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
		mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

		//Add to Map
		mpMap->AddMapPoint(pMP);
	}

	// Update Connections
	pKFini->UpdateConnections();
	pKFcur->UpdateConnections();

	// Bundle Adjustment
	cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

	Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

	// Set median depth to 1
	float medianDepth = pKFini->ComputeSceneMedianDepth(2);
	float invMedianDepth = 1.0f / medianDepth;

	if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
	{
		cout << "Wrong initialization, reseting..." << endl;
		Reset();
		return;
	}

	// Scale initial baseline
	cv::Mat Tc2w = pKFcur->GetPose();
	Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3)*invMedianDepth;
	pKFcur->SetPose(Tc2w);

	// Scale points
	vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
	for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
	{
		if (vpAllMapPoints[iMP])
		{
			MapPoint* pMP = vpAllMapPoints[iMP];
			pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
		}
	}

	mpLocalMapper->InsertKeyFrame(pKFini);
	mpLocalMapper->InsertKeyFrame(pKFcur);

	mCurrentFrame.SetPose(pKFcur->GetPose());
	mnLastKeyFrameId = mCurrentFrame.mnId;
	mpLastKeyFrame = pKFcur;

	mvpLocalKeyFrames.push_back(pKFcur);
	mvpLocalKeyFrames.push_back(pKFini);
	mvpLocalMapPoints = mpMap->GetAllMapPoints();
	mpReferenceKF = pKFcur;
	mCurrentFrame.mpReferenceKF = pKFcur;

	mLastFrame = Frame(mCurrentFrame);

	mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

	mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

	mpMap->mvpKeyFrameOrigins.push_back(pKFini);

	mState = OK;
}

//if a MapPoint was replaced it updates the MapPoint list. For some reason we don't delete mappoints, we just load the replacement to a mappoints mpReplacement pointer.
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
	std::stringstream ss_log;
	// Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
	if (orb_prop_log)
	{
		ss_log << "\tMatches after SearchByBoW on RefKF: " << nmatches << "\n";
		helperFunctions::log(ss_log, MyObjects->logPath);
	}
    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

	mnMatchesInliers = 0;
	for (int i = 0; i < mCurrentFrame.N; i++)
	{
		if ((mCurrentFrame.mvpMapPoints[i]) && (!mCurrentFrame.mvbOutlier[i]) && (mCurrentFrame.mvpMapPoints[i]->Observations() > 0))
			mnMatchesInliers++;
	}
	if (orb_prop_log)
	{
		ss_log << "\tMatches after PoseOptimization: " << mnMatchesInliers << "\n";
		helperFunctions::log(ss_log, MyObjects->logPath);
	}
	int outliers = 0;
    // Discard outliers

	int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
				outliers++;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
	if (orb_prop_log)
	{
		ss_log << "\t\tClassified as outliers after PoseOptimization(): " << outliers << "\n";
		helperFunctions::log(ss_log, MyObjects->logPath);
	}
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

	//We always return here, because we only use the monocular part.
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
	std::stringstream ss_log;
	ORBmatcher matcher(0.9,true);
	
	std::cout<< "Motion model poseInverse: \n" << mLastFrame.mTcw.inv() << "\n";

	if (orb_prop_log)
	{
		ss_log << "TrackWithMotionModel() matches:\n";
		helperFunctions::log(ss_log, MyObjects->logPath);
	}
    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
	
	//It updates the location of the last frame. The frame locations are stored as relative coordinate transformations to their reference KF-s. 
	//So if the RefKF-s pose is updated, we have to update the Frame's pose too.
	UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));


    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR,MyObjects->logPath);


    // If few matches, uses a wider window search
    if(nmatches<20)
    {
		ss_log << "Too few matches first time. SearchByProjection again with double the treshold for error:\n";
		helperFunctions::log(ss_log, MyObjects->logPath);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR, MyObjects->logPath);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);
	if (orb_prop_log)
	{
		ss_log << "\tPoseOptimization():\n";
		helperFunctions::log(ss_log, MyObjects->logPath);
	}
    // Discard outliers
	int outliers = 0;
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
				outliers++;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

	if (orb_prop_log)
	{
		ss_log << "\t\tClassified as outliers after PoseOptimization(): " << outliers;
		helperFunctions::log(ss_log, MyObjects->logPath);
	}
    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    UpdateLocalMap();

    SearchLocalPoints();

    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently

    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
	stringstream ss_log;

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);//number of points with minimum nMinObs observations.

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
	//because I use monocular, here this is false.
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }
	//it will be false
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

	//I use this
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
	//makes no sense because minframes is 0.
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
	//it is always false, because Monocular
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

	
	ss_log << "\t\tc1a: " << c1a << " c1b: " << c1b << " c1c: " << c1c << " c2: " << c2 << "   \r";
	helperFunctions::log(ss_log, MyObjects->logPath);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    //if mapping hasn't stopped
	if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;
	
	//I only use monocular
    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;

	//I increment the variable where I store how many keyframes were added since last NMI relocalization
	keyFramesSinceLastNmi++;

}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{
	std::stringstream ss_log;

	ss_log << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }
    // Reset Local Mapping
	ss_log << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
	ss_log << " done" << endl;
	helperFunctions::log(ss_log, MyObjects->logPath);

    // Reset Loop Closing
	ss_log << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
	ss_log << " done" << endl;
	helperFunctions::log(ss_log, MyObjects->logPath);

    // Clear BoW Database
	ss_log << "Reseting Database...";
    mpKeyFrameDB->clear();
	ss_log << " done" << endl;
	helperFunctions::log(ss_log, MyObjects->logPath);

    // Clear Map (this erases MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

//NMI Relocalization
//It uses the KF or F position and the map to calculate the refined location of the KF or F.
//It updates the loaction of the KF or F.
void Tracking::RelocalizeWithNMI(void *frame, bool isKF) {
	stringstream ss_log;

	cv::Mat Tcw;
	cv::Mat Twc;
	int Id;
	if (isKF) {
		Tcw = static_cast<KeyFrame*>(frame)->GetPose();
		Twc = static_cast<KeyFrame*>(frame)->GetPoseInverse();
		Id = static_cast<KeyFrame*>(frame)->mnId;
	}
	else {
		Tcw = static_cast<Frame*>(frame)->GetPose();
		Twc = static_cast<Frame*>(frame)->GetPoseInverse();
		Id = static_cast<Frame*>(frame)->mnId;
	}


	char end;
	//should start a new thread
	MyImage->loadOriginal(this->mImGray.clone());
	MyImage->calculateWarping();
	CameraSettings settings = setupCam(Twc, MyImage->getK());

	//sets up camera
	MyRenderer->setCamera(settings.getPosition(), settings.getDirection(), settings.getUp());
	int itmax = MyImage->getNumWarpX()*MyImage->getNumWarpY()*MyImage->getNumWarpZ()*MyRenderer->getNumSynthX()*MyRenderer->getNumSynthY()*MyRenderer->getNumSynthZ();
	int iterator = 0;
	for (int sX = 0; sX < MyRenderer->getNumSynthX(); sX++) {
		for (int sY = 0; sY < MyRenderer->getNumSynthY(); sY++) {
			for (int sZ = 0; sZ < MyRenderer->getNumSynthZ(); sZ++) {
				MyRenderer->renderToTextureOnGPU(MyRenderer->calculateTranslation(sX, sY, sZ));
				for (int wX = 0; wX < MyImage->getNumWarpX(); wX++) {
					for (int wY = 0; wY < MyImage->getNumWarpY(); wY++) {
						for (int wZ = 0; wZ < MyImage->getNumWarpZ(); wZ++) {
							CUDAF::NMIWithCuda_noMask(
								(cv::cuda::PtrStep<unsigned char>*)MyImage->getImageGPU(wZ, wY, wX).data,
								SUC,
								MATCHING_NMI,
								MyRenderer->getImageWidth(),
								MyRenderer->getImageHeight(),
								&(MyObjects->rating[wZ][wY][wX][sZ][sY][sX]),
								MyObjects->myRenderer->getrenderedTexture()
							);
							std::cout << iterator + 1 << "/" << itmax << "      \r";
							iterator++;
						}
					}
				}
			}
		}
	}
	//find the extremeelements
	std::vector<NmiSearchKernel> ExtremElements;
	ExtremElements = helperFunctions::find_max_elements(MyObjects->rating, *MyObjects->NmiKernel);



	//save the nice pictures with every info
	//extremeElements should be a vector with only one element
	if (orb_prop_log){
		for (NmiSearchKernel ExtremElement : ExtremElements)
		{
			std::stringstream ss;
			//max 260 character in the full path(windows)
			ss.precision(5);
			ss << std::fixed;
			ss <<
				MyObjects->resultsPath << "/" << "id_" << Id << "_s_zyx_[" <<
				MyObjects->NmiKernel->getNumSynthZ() << "x" << std::setw(7) << MyObjects->NmiKernel->getStepZ() << "_m]_[" <<
				MyObjects->NmiKernel->getNumSynthY() << "x" << std::setw(7) << MyObjects->NmiKernel->getStepY() << "_m]_[" <<
				MyObjects->NmiKernel->getNumSynthX() << "x" << std::setw(7) << MyObjects->NmiKernel->getStepX() << "_m]_w_zyx_[" <<
				MyObjects->NmiKernel->getNumWarpZ() << "x" << std::setw(7) << MyObjects->NmiKernel->getStepRadZ() << "_rad]_[" <<
				MyObjects->NmiKernel->getNumWarpY() << "x" << std::setw(7) << MyObjects->NmiKernel->getStepRadY() << "_rad]_[" <<
				MyObjects->NmiKernel->getNumWarpX() << "x" << std::setw(7) << MyObjects->NmiKernel->getStepRadX() << "_rad]_NMI_[" <<
				ExtremElement.getNmi() << "]_WzyxSzyx_[" <<
				ExtremElement.getBestWarpZ() << "," << ExtremElement.getBestWarpY() << "," << ExtremElement.getBestWarpX() << "," <<
				ExtremElement.getBestSynthZ() << "," << ExtremElement.getBestSynthY() << "," << ExtremElement.getBestSynthX() << "].jpg";
			cv::Mat ImToSave;
			MyImage->getImageGPU(ExtremElement.getBestWarpZ(), ExtremElement.getBestWarpY(), ExtremElement.getBestWarpX()).download(ImToSave);


			std::vector<unsigned char> best, mid;
			best.resize(MyRenderer->getImageHeight()*MyRenderer->getImageWidth());
			mid.resize(MyRenderer->getImageHeight()*MyRenderer->getImageWidth());
			MyRenderer->renderToTexture(MyRenderer->calculateTranslation(ExtremElement.getBestSynthX(), ExtremElement.getBestSynthY(), ExtremElement.getBestSynthZ()), best);
			MyRenderer->renderToTexture(MyRenderer->calculateTranslation(MyObjects->NmiKernel->getNumSynthZ() / 2, MyObjects->NmiKernel->getNumSynthY() / 2, MyObjects->NmiKernel->getNumSynthX() / 2), mid);

			saveImage(
				(ss.str()).c_str(), //first param is the name of the file
				ImToSave, //warped image
				best,//best synthetic match
				MyImage->getOriginal(), //original image
				mid//middle synthetic
			);
			helperFunctions::log(ss_log, MyObjects->logPath);
		}
	}
	

	//setting the best match in the NmiKernel
	MyObjects->NmiKernel->setBest(&ExtremElements[0]);
	MyObjects->NmiKernel->NMI = ExtremElements[0].getNmi();

	//acquire the relocalized Location
	cv::Mat newLoc = CalculateNMIRelocalization(Twc, *MyObjects->NmiKernel, *MyRenderer);
	
	if (orb_prop_log) {
		//set the newly calculated pose as the camera pose
		settings = setupCam(newLoc, MyImage->getK());
		MyRenderer->setCamera(settings.getPosition(), settings.getDirection(), settings.getUp());

		std::stringstream ss;
		ss << MyObjects->resultsPath << "/" << Id << "_renderedFromNewPos_nmi[" << MyObjects->NmiKernel->NMI << "].jpg";
		//render the middle to mid
		std::vector<unsigned char> rendered;
		rendered.resize(MyRenderer->getImageHeight()*MyRenderer->getImageWidth());
		MyRenderer->renderToTexture(MyRenderer->calculateTranslation(MyObjects->NmiKernel->getNumSynthZ() / 2, MyObjects->NmiKernel->getNumSynthY() / 2, MyObjects->NmiKernel->getNumSynthX() / 2), rendered);

		//save the image
		saveImage((ss.str()).c_str(),
			MyImage->getOriginal(),
			rendered);
	}

	if (isKF){
		static_cast<KeyFrame*>(frame)->SetPose(newLoc.inv());
		static_cast<KeyFrame*>(frame)->SetNMIRelocalized(true);
	}
	else{
		static_cast<Frame*>(frame)->SetPose(newLoc.inv());
		static_cast<Frame*>(frame)->SetNMIRelocalized(true);
	}
	
};

void Tracking::RelocalizeWithNMIStrategy(void *frame, bool isKF)
{
	std::stringstream ss_log;
	bool end = false;
	int i = 0;
	int underTreshold = 0;
	bool lastWasUnderTreshold = false;
	cv::Mat TcwSave, TcwSaveLast;

	//reset everything
	MyObjects->NmiKernel->reset();
	MyObjects->LastNmiKernel->reset();
	//set the Kernel size (if distance is 0, than we set the initials)

	if (mDistanceSinceLastNMI(0) > 0.0) {
		//the new step sizes depend on the distance since the last nmi relocalization.
		//ORB-SLAM 2 has a drift about 1% so we conservatively search in the 2% proximity
		float step_x = mDistanceSinceLastNMI(0) * 0.02;
		float step_y = mDistanceSinceLastNMI(1) * 0.02;
		float step_z = mDistanceSinceLastNMI(2) * 0.02;

		float step_rad_x = mRotationSinceLastNMI(0) * 0.02;
		float step_rad_y = mRotationSinceLastNMI(1) * 0.02;
		float step_rad_z = mRotationSinceLastNMI(2) * 0.02;

		int num_x, num_y, num_z, num_warp_x, num_warp_y, num_warp_z;
		
		if (step_x < nmi_prop_MIN_KERNEL_TRANSLATION)
			num_x = 1;
		else
			num_x = MyObjects->InitialNmiKernel->getNumSynthX();

		if (step_y < nmi_prop_MIN_KERNEL_TRANSLATION)
			num_y = 1;
		else
			num_y = MyObjects->InitialNmiKernel->getNumSynthY();

		if (step_z < nmi_prop_MIN_KERNEL_TRANSLATION)
			num_z = 1;
		else
			num_z = MyObjects->InitialNmiKernel->getNumSynthZ();


		if (step_rad_x < nmi_prop_MIN_KERNEL_ROTATION)
			num_warp_x = 1;
		else
			num_warp_x = MyObjects->InitialNmiKernel->getNumWarpX();

		if (step_rad_y < nmi_prop_MIN_KERNEL_ROTATION)
			num_warp_y = 1;
		else
			num_warp_y = MyObjects->InitialNmiKernel->getNumWarpY();

		if (step_rad_z < nmi_prop_MIN_KERNEL_ROTATION)
			num_warp_z = 1;
		else
			num_warp_z = MyObjects->InitialNmiKernel->getNumWarpZ();

		//set the searckernel
		MyObjects->NmiKernel->setKernel(
			num_x, num_y, num_z,
			num_warp_x, num_warp_y, num_warp_z,	
			step_x, step_y, step_z,
			step_rad_x, step_rad_y, step_rad_z);
		//update NMI objects
		MyObjects->setNmiObjectsKernel(MyObjects->NmiKernel);
	}
	else {
		if (mState == NOT_INITIALIZED) {//during initialization we use a biger kernel in the translational freedooms
			//set the searckernel
			MyObjects->NmiKernel->setKernel(5, 5, 5,
				MyObjects->InitialNmiKernel->getNumWarpX(), MyObjects->InitialNmiKernel->getNumWarpY(), MyObjects->InitialNmiKernel->getNumWarpZ(),
				MyObjects->InitialNmiKernel->getStepX(), MyObjects->InitialNmiKernel->getStepY(), MyObjects->InitialNmiKernel->getStepZ(),
				MyObjects->InitialNmiKernel->getStepRadX(), MyObjects->InitialNmiKernel->getStepRadY(), MyObjects->InitialNmiKernel->getStepRadZ());
			//update NMI objects
			MyObjects->setNmiObjectsKernel(MyObjects->NmiKernel);
		}
		else {
			//set the searckernel
			MyObjects->NmiKernel->setKernel(MyObjects->InitialNmiKernel);
			//update NMI objects
			MyObjects->setNmiObjectsKernel(MyObjects->NmiKernel);
		}
	}

	if (orb_prop_log)
	{
		if (isKF)
			ss_log << "\n----------------//\\\\----------------\nKeyFrame PoseInverse before NMI relocalization:\n" << static_cast<KeyFrame*>(frame)->GetPoseInverse() << "\n----------------//\\\\----------------\n";
		else
			ss_log << "\n----------------//\\\\----------------\nFrame PoseInverse before NMI relocalization:\n" << static_cast<Frame*>(frame)->GetPoseInverse() << "\n----------------//\\\\----------------\n";
		helperFunctions::log(ss_log, MyObjects->logPath);
	}

	if (isKF)
		TcwSave = static_cast<KeyFrame*>(frame)->GetPose().clone();
	else
		TcwSave = static_cast<Frame*>(frame)->GetPose().clone();

	TcwSaveLast = TcwSave.clone();

	while (!end)
	{
		i++;
		if (i > nmi_prop_MAX_ITERATION_COUNT)//don't loop forever
			break;

		if (isKF)
			static_cast<KeyFrame*>(frame)->mvPreviousPoses.push_back(static_cast<KeyFrame*>(frame)->GetPoseInverse());
		else
			static_cast<Frame*>(frame)->mvPreviousPoses.push_back(static_cast<Frame*>(frame)->GetPoseInverse());
	

 		RelocalizeWithNMI(frame, isKF);

		
		ss_log << "NmiKernel:\t" << *MyObjects->NmiKernel;
		ss_log << "\nLastNmiKernel:\t" << *MyObjects->LastNmiKernel;
		ss_log << "\nKernel rate:\t" << (MyObjects->NmiKernel->getNmi() / MyObjects->LastNmiKernel->getNmi()) << "\n";
		helperFunctions::log(ss_log, MyObjects->logPath);
	
		if (i > 1)
			if (MyObjects->NmiKernel->isMiddle())
				break;

		if (i > 1) {
			if ((MyObjects->NmiKernel->getNmi() / MyObjects->LastNmiKernel->getNmi()) < 1.001){
				if (underTreshold > 0)
					break;
				else
					underTreshold++;
			}
			else
				underTreshold = 0;
		}

		//recalculates the nmiSearchKernel and than updates the nmi objects
		MyObjects->NMIobjectsReInitialization();

		if (isKF)
			TcwSaveLast = static_cast<KeyFrame*>(frame)->GetPose().clone();
		else
			TcwSaveLast = static_cast<Frame*>(frame)->GetPose().clone();
	}


	//if exited because of lower NMI than last
	if (MyObjects->NmiKernel->getNmi() < MyObjects->LastNmiKernel->getNmi()){
		if (isKF)
			static_cast<KeyFrame*>(frame)->SetPose(TcwSaveLast);
		else
			static_cast<Frame*>(frame)->SetPose(TcwSaveLast);
	}
	//the NMI treshold is dependent on the distance since the last NMI. If the distance is greater than NMI_baseline the treshold will be smaller than the preset.
	//This way we can accept worse relocalization and giving the algorithm a chance to recover at the next 
	//it basically lets the algorithm to accept worse than optimal results in order to not lose nmi relocalization capabilities
	double NMI_treshold;
	double NMI_baseline = 5;
	double distanceSinceLastNMI = sqrt(pow(mDistanceSinceLastNMI(0),2)+ pow(mDistanceSinceLastNMI(1), 2)+ pow(mDistanceSinceLastNMI(2), 2));
	if (distanceSinceLastNMI < NMI_baseline)
		NMI_treshold = mfNmiInitTresholf;
	else{
		NMI_treshold = mfNmiInitTresholf * (NMI_baseline / distanceSinceLastNMI);
		if (NMI_treshold < (mfNmiInitTresholf / 2))
			NMI_treshold = mfNmiInitTresholf / 2;
	}


		

	if (MyObjects->NmiKernel->getNmi()< NMI_treshold){
		if (isKF){
			static_cast<KeyFrame*>(frame)->SetPose(TcwSave);
			static_cast<KeyFrame*>(frame)->SetNMIRelocalized(false);
			static_cast<KeyFrame*>(frame)->SetNMIFailed(true);
		}
		else{
			static_cast<Frame*>(frame)->SetPose(TcwSave);
			static_cast<Frame*>(frame)->SetNMIRelocalized(false);
			static_cast<Frame*>(frame)->SetNMIFailed(true);
		}
	}

	if (orb_prop_log){
		if (isKF){
			ss_log << "\n----------------//\\\\----------------\nKeyFrame PoseInverse after NMI relocalization:\n" << static_cast<KeyFrame*>(frame)->GetPoseInverse() << "\n----------------//\\\\----------------\n";
		}
		else{
			ss_log << "\n----------------//\\\\----------------\nFrame PoseInverse after NMI relocalization:\n" << static_cast<Frame*>(frame)->GetPoseInverse() << "\n----------------//\\\\----------------\n";
		}
		helperFunctions::log(ss_log, MyObjects->logPath);
	}	
}

void Tracking::InitializeWithNMI()
{
	std::stringstream ss_log;
	if (mInitialFrame.mTcw.empty())
	{		
		mCurrentFrame.SetPose(mInit1);
		RelocalizeWithNMIStrategy(&mCurrentFrame, false);
		mInitialFrame = mCurrentFrame;
	}
	else
	{
		mCurrentFrame.SetPose(mInit2);
		RelocalizeWithNMIStrategy(&mCurrentFrame, false);

		// Create KeyFrames
		KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
		KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

		//we add the first KF to the relative frame poses list, so when we save the trajectory it will also be saved.
		//The second will be added automatically, because the state will be OK.
		mlRelativeFramePoses.push_back(cv::Mat::eye(4, 4, CV_32F));
		mlFrameNumberList.push_back(0);
		mlpReferences.push_back(pKFini);
		mlFrameTimes.push_back(mInitialFrame.mTimeStamp);
		mlbLost.push_back(mState == LOST);

		pKFini->ComputeBoW();
		pKFcur->ComputeBoW();

		// Insert KFs in the map
		mpMap->AddKeyFrame(pKFini);
		mpMap->AddKeyFrame(pKFcur);

		std::vector<int> initMatches;

		//Find correspondences
		ORBmatcher matcher(0.9, true); 


		int matches = matcher.SearchByBoW(pKFini, pKFcur, initMatches);

		ss_log << "-----------------------------------------\n\nInitial matches from SearchByBoW:" << matches << "/" << initMatches.size() << "\n\n-----------------------------------------\n";
		helperFunctions::log(ss_log, MyObjects->logPath);


		int wierdos = 0;
		int negativeZ1 = 0;
		int negativeZ2 = 0;

		int MPcount = 0;
		// Camera Projection Matricies K[R|t]
		cv::Mat	P1 = pKFini->mK.clone()*pKFini->GetPose().rowRange(0, 3).colRange(0, 4);
		cv::Mat P2 = pKFini->mK.clone()*pKFcur->GetPose().rowRange(0, 3).colRange(0, 4);

		for (size_t i = 0; i < initMatches.size(); i++)
		{
			if (initMatches[i] > 0)
			{
				bool wierd1 = false;
				//KPs in Init and Current KF
				const cv::KeyPoint &kp1 = pKFini->mvKeysUn[i];
				const cv::KeyPoint &kp2 = pKFcur->mvKeysUn[initMatches[i]];
				cv::Mat p3d;
				
				Initializer::Triangulate(kp1, kp2, P1, P2, p3d);

				//every coordinate should be finite
				if (!isfinite(p3d.at<float>(0)) || !isfinite(p3d.at<float>(1)) || !isfinite(p3d.at<float>(2)))
				{
					continue;
				}

				// Check parallax
				cv::Mat normal1 = p3d - pKFini->GetCameraCenter();
				float dist1 = cv::norm(normal1);

				cv::Mat normal2 = p3d - pKFcur->GetCameraCenter();
				float dist2 = cv::norm(normal2);

				float cosParallax = normal1.dot(normal2) / (dist1*dist2);

				//The points in the camera frame coordinate system
				cv::Mat p3d_h = toHomogeneous(p3d);

				cv::Mat p3dC1 = fromHomogeneous(pKFini->GetPose()*p3d_h);
				cv::Mat p3dC2 = fromHomogeneous(pKFcur->GetPose()*p3d_h);

				// Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
				if (p3dC1.at<float>(2) <= 0)
				{
					negativeZ1++;
					if (cosParallax < 0.99998)
						continue;
					else
					{
						wierd1 = true;
						continue;//maybe shouldn't continue here
					}
				}
				// Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
				if (p3dC2.at<float>(2) <= 0)
				{
					negativeZ2++;
					if (cosParallax < 0.99998)
						continue;
					else
					{
						if (wierd1)
							wierdos++;
						continue;
					}
				}
				// Check reprojection error in first image
				float im1x, im1y;
				float invZ1 = 1.0 / p3dC1.at<float>(2);
				im1x = pKFini->fx * p3dC1.at<float>(0)*invZ1 + pKFini->cx;
				im1y = pKFini->fy * p3dC1.at<float>(1)*invZ1 + pKFini->cy;

				float squareError1 = (im1x - kp1.pt.x)*(im1x - kp1.pt.x) + (im1y - kp1.pt.y)*(im1y - kp1.pt.y);

				if (squareError1 > 4.0)
					continue;

				// Check reprojection error in second image
				float im2x, im2y;
				float invZ2 = 1.0 / p3dC2.at<float>(2);
				im2x = pKFini->fx * p3dC2.at<float>(0)*invZ2 + pKFini->cx;
				im2y = pKFini->fy * p3dC2.at<float>(1)*invZ2 + pKFini->cy;

				float squareError2 = (im2x - kp2.pt.x)*(im2x - kp2.pt.x) + (im2y - kp2.pt.y)*(im2y - kp2.pt.y);

				if (squareError2 > 4.0)
					continue;

				MPcount++;

				MapPoint* pMP = new MapPoint(p3d, pKFcur, mpMap);

				pKFini->AddMapPoint(pMP, i);
				pKFcur->AddMapPoint(pMP, initMatches[i]);

				pMP->AddObservation(pKFini, i);
				pMP->AddObservation(pKFcur, initMatches[i]);

				pMP->ComputeDistinctiveDescriptors();
				pMP->UpdateNormalAndDepth();

				//Fill Current Frame structure
				mCurrentFrame.mvpMapPoints[initMatches[i]] = pMP;
				mCurrentFrame.mvbOutlier[initMatches[i]] = false;

				//Add to Map
				mpMap->AddMapPoint(pMP);
			}
		}
		
		ss_log << "\nInitial MapPoint Count: " << MPcount<<"\n";
		helperFunctions::log(ss_log, MyObjects->logPath);

		pKFini->UpdateConnections();
		pKFcur->UpdateConnections();

		// Bundle Adjustment
		Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

		mpLocalMapper->InsertKeyFrame(pKFini);//adds the KF to the end of the newKeyFrames vector
		mpLocalMapper->InsertKeyFrame(pKFcur);

		mnLastKeyFrameId = mCurrentFrame.mnId;
		mpLastKeyFrame = pKFcur;

		mvpLocalKeyFrames.push_back(pKFcur);
		mvpLocalKeyFrames.push_back(pKFini);
		mvpLocalMapPoints = mpMap->GetAllMapPoints();
		mpReferenceKF = pKFcur;
		mCurrentFrame.mpReferenceKF = pKFcur;

		mLastFrame = Frame(mCurrentFrame);

		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

		mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

		mpMap->mvpKeyFrameOrigins.push_back(pKFini);

		mState = OK;

		std::stringstream path;
		path << MyObjects->resultsPath << "/OriginalMap.xyz";
		PrintMapPoints(path.str(), 127, 255, 127);
	}
}

cv::Mat Tracking::CalculateNMIRelocalization(cv::Mat Twc, NmiSearchKernel &nmiKernel, Rendering<nmi_prop_RENDER> &renderer)
{
	//acquire the relocalized Location
	cv::Mat newLoc = cv::Mat::eye(4, 4, CV_32F);
	cv::Mat oldLoc = Twc.clone();
	cv::Mat NMITransform = cv::Mat::eye(4, 4, CV_32F);
	cv::Mat translation = cv::Mat(3, 1, CV_32F);

	//rotation angles
	float rotx = (nmiKernel.getBestWarpX() - (nmiKernel.getNumWarpX() / 2))*nmiKernel.getStepRadX();
	float roty = (nmiKernel.getBestWarpY() - (nmiKernel.getNumWarpY() / 2))*nmiKernel.getStepRadY();
	float rotz = (nmiKernel.getBestWarpZ() - (nmiKernel.getNumWarpZ() / 2))*nmiKernel.getStepRadZ();

	cv::Mat R, Rx, Ry, Rz;

	//Rotation about x (points right)
	Rx = (cv::Mat_<float>(3, 3) <<
		1, 0, 0,
		0, cos(rotx), -sin(rotx),
		0, sin(rotx), cos(rotx));

	//Rotation about y (points ?down?)
	Ry = (cv::Mat_<float>(3, 3) <<
		cos(roty), 0, sin(roty),
		0, 1, 0,
		-sin(roty), 0, cos(roty));

	//Rotation about z (points ?forwards?)
	Rz = (cv::Mat_<float>(3, 3) <<
		cos(rotz), -sin(rotz), 0,
		sin(rotz), cos(rotz), 0,
		0, 0, 1);

	R = Rz*Ry*Rx;

	R.copyTo(NMITransform(cv::Rect(0, 0, 3, 3)));

	newLoc = Twc * NMITransform;
	translation = renderer.calculateTranslationCV(nmiKernel.getBestSynthX(), nmiKernel.getBestSynthY(), nmiKernel.getBestSynthZ());

	newLoc.at<float>(0, 3) += translation.at<float>(0, 0);
	newLoc.at<float>(1, 3) += translation.at<float>(1, 0);
	newLoc.at<float>(2, 3) += translation.at<float>(2, 0);

	return newLoc;
}

//Other useful functions to save some data during testing
void  Tracking::PrintMapPoints(std::string FileName, int r, int g, int b)
{
	const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
	vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
	
	ofstream myfile, myfile2;

	std::string FName = FileName;
	FName.erase(FName.end() - 4, FName.end());
	FName.append("_onlyMPs.xyz");

	myfile.open(FileName);
	myfile2.open(FName);

	myfile << "MPs\n";
	for (int i = 0; i < vpMPs.size(); i++)
	{
		myfile<< vpMPs[i]->GetWorldPos().at<float>(0) << ", " << vpMPs[i]->GetWorldPos().at<float>(1) << ", " << vpMPs[i]->GetWorldPos().at<float>(2) << "\n";
		myfile2 << vpMPs[i]->GetWorldPos().at<float>(0) << ", " << vpMPs[i]->GetWorldPos().at<float>(1) << ", " << vpMPs[i]->GetWorldPos().at<float>(2) << ", " << r<<", "<<g<< ", " << b<<"\n";
	}
	myfile << "KFs\n";
	for (int i = 0; i < vpKFs.size(); i++)
	{
		myfile<< vpKFs[i]->GetPoseInverse() << "\n";
	}
	myfile.close();
	myfile2.close();
}

void Tracking::AppendMatToFile(std::string FileName, cv::Mat mat)
{
	std::ofstream output;
	output.open(FileName, std::ios_base::app);
	output << mat << "\n";
	output.close();
}

} //namespace ORB_SLAM
