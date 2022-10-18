# orbslam2_NMI

This is a windows build for orbslam2_NMI. We release the code under GNU licence, but for licence information please check LICENSE.txt.

## Additional material

A long video demonstration of the results of the algorithm can be found here: \
https://youtu.be/HLvOK9Uwykk \
A shorter summary can be found here: \
https://youtu.be/MHC67S7Vvt4

As a contribution we share the 3D mesh and 3D pointcloud calculated for the The Zurich Urban Micro Aerial Vehicle Dataset (ZU-MAV). \
The dataset and the publication can be found here:\
http://rpg.ifi.uzh.ch/zurichmavdataset.html \
Our contribution (mesh and pointcloud model) can be found here:\
https://drive.google.com/drive/folders/1mLt51wLT-8Aqe7TcGoL3ARhLTnubnck0?usp=sharing\

## Download and build instructions

This code is built for windows, although all the underlying dependencies have linux versions, so it should be possible to build it on Linux.

**1.** Clone the repo.

**2.** You can build the different dependencies, or you can download the prebuilt binarys from here:\
https://drive.google.com/file/d/1Xwr5CBOHRDwqndI2jQ4nJKdRiY3eZr2s/view?usp=sharing
After download the following folder structere is assumed:\
somefolder\
├── Externals\
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Cuda \
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── OpenCV \
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── etc. \
├── somotherfolder\
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── orbslam2_NMI \
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── build \
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Examples \
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── etc. \
If you want to use other structure you will have to specify the lib and header locations in the project files.

If you chose to build the dependencies, you can find the versions in the above zip file. If you choose this road you will have to specify the lib file and header file locations for visual studio in the project properties.

**3.** When opening the solution use the set up toolsets, don't change them. We used Visual Studio 2017 but newer versions shouldn't be a problem.

**4.** Set mono_ETH or mono_newer_college as starter projects

**5.** If the dependencies are in the correct location you can build the solution

**6.** To run the solution you will need opencv_world.dll. 
If you downloaded the dependencies, you can find it here: opencv-3.4.0_CUDA\build\install\x64\vc15\bin
copy it to the orbslam2_NMI\Examples\Monocular\Release folder

**7.** you will need to create a folder for the results. 
In the localization project, there is a header file called allProperties.hpp. In that file, there is a defined value called nmi_prop_OUTPUT_LOC. This can be anything (absolute or relative path), but it has to exist. The results will be generated into this folder.

**8.** Now you can run the algorithm. If you run it from visual studio, with local windows debugger you can specify the command arguments in the project settings. If you run the built exe file you should run it from cmd with arguments. Usage is as follows:
Usage: ./mono_ETH path_to_vocabulary path_to_settings path_to_sequence 

**9.** In the vocabulary folder there is a ORBvoc.txt.tar.gz file. You have to unzip it to get ORBvoc.txt. path_to_vocabulary will be ../Vocabulary/ORBvoc.txt

**10.** You will have to specify the yaml file. The dataset locations and other properties are in these files, so those path-s should be updated. The yaml files are in orbslam2_NMI\Examples\Monocular If you specified everything correctly path_to_settings will look something like this ../Examples/Monocular/ETH_small.yaml, and the path_to_sequence will have to point to the folder where the images are.

**11.** If you are using your own dataset you will have to modify the LoadImages function to load the correct timestamps and image names for the sequence.

If you have any questions: <gazdag.sandor at sztaki dot hu>

## Datasets

We tested the algorithm on publicly avaiable datasets. For the ZU-MAV dataset the map of the environment is published above. The camera images can be dowwnloaded from here:\
http://rpg.ifi.uzh.ch/zurichmavdataset.html \
We tested the algorithm for the Newer College dataset. That can be found here:\
https://ori-drs.github.io/newer-college-dataset/ \
## Special thanks

Special thanks to Raul Mur-Artal, Juan D. Tardos, J. M. M. Montiel and Dorian Galvez-Lopez (DBoW2) for ORB-SLAM2:\
https://github.com/raulmur/ORB_SLAM2 \
And to phdsky for the instructions on windows:\
https://github.com/phdsky/ORBSLAM24Windows
