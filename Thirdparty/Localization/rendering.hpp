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

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <glfw3.h>

//Include OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "helperFunctions.hpp"
#include "allProperties.hpp"

#define RENDER_TEXTURE 1
#define RENDER_POINT_CLOUD 4

#define TESTING_IMAGES 0
#define TESTING_NO_IMAGES 1

//for memtest
#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

template <unsigned char RenderingMode>
class Rendering
{
	int windowWidth;
	int windowHeight;
	int imageWidth;
	int imageHeight;
	int numSynthX;
	int numSynthY;
	int numSynthZ;
	float stepX;
	float stepY;
	float stepZ;

	float PointSize;

	int current_level;

	GLuint programID;
	GLuint MatrixID;

	GLuint Texture;
	GLuint TextureID;

	std::vector<glm::vec3> cloud_vertices;
	std::vector<glm::vec3> colors;
	std::vector<glm::vec2> uvs;

	GLuint cloud_vertexbuffer;
	GLuint colorbuffer;
	GLuint uvbuffer;

	GLuint FramebufferName;
	GLuint renderedTexture;
	GLuint depthrenderbuffer;

	GLenum DrawBuffers[1];
	GLfloat g_quad_vertex_buffer_data[18];
	GLuint quad_vertexbuffer;

	GLuint quad_programID;

	GLuint VertexArrayID;

	glm::mat4 Projection;
	glm::mat4 ViewMatrix;
	glm::mat4 MVP;
	glm::vec3 Camera_pos;
	glm::vec3 Camera_direction;
	glm::vec3 Camera_up;

	GLint part_width;
	GLint part_height;

	std::string logPath;
public: 
	bool CreateFrameBuffers();
	
	bool initGL();
	
	bool initVBO(double near_clipping_plane, double far_clipping_plane, double fx, double fy, double cx, double cy, std::string object_path, std::string texture_path, std::string cloud_path, std::string offset_path);

	bool initQuad();

	Rendering(const float& pointSize, const int&windowwidth, const int&windowheight, const int&imagewidth, const int&imageheight, const int &num_of_views_x, const int & num_of_views_y, const int & num_of_views_z, const float &stepx, const float & stepy, const float & stepz, const glm::vec3 &Cam_pos, const glm::vec3 &Cam_dir, const glm::vec3 &Cam_up, double near_clipping_plane, double far_clipping_plane, double fx, double fy, double cx, double cy, std::string object_path, std::string texture_path, std::string cloud_path, std::string offset_path, std::string log_Path);

	bool renderToTexture(const glm::vec3& translation, std::vector<unsigned char> &synth);

	bool renderToTextureOnGPU(const glm::vec3& translation);

	bool setCamera(const glm::vec3 &Cam_pos, const glm::vec3 &Cam_dir, const glm::vec3 &Cam_up);

	glm::vec3 calculateTranslation(int synthx, int synthy, int synthz);

	cv::Mat calculateTranslationCV(int synthx, int synthy, int synthz);

	~Rendering();

	void resizeKernel(const int &num_of_views_x, const int & num_of_views_y, const int & num_of_views_z, const float &stepx, const float & stepy, const float & stepz);

	//get functions
	int getImageWidth();
	int getImageHeight();
	int getNumSynthX();
	int getNumSynthY();
	int getNumSynthZ();
	float getStepX();
	float getStepY();
	float getStepZ();
	
	unsigned int getFramebufferName();
	unsigned int getrenderedTexture();

	void setSynthetic_count_x(int sx);
	void setSynthetic_count_y(int sy);
	void setSynthetic_count_z(int sz);
	void setStep_x(float stx);
	void setStep_y(float sty);
	void setStep_z(float stz);
};

//Vertex Buffer Object Initialization. Constructor uses it.
template <unsigned char RenderingMode>
bool Rendering<RenderingMode>::initVBO(double near_clipping_plane, double far_clipping_plane, double fx, double fy, double cx, double cy, std::string object_path, std::string texture_path, std::string cloud_path, std::string offset_path)
{
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);
	//load shaders
	switch (RenderingMode)
	{
	case RENDER_POINT_CLOUD:
			programID = LoadShaders(nmi_prop_SHADER_LOC "ShadingWithColor.vertexshader", nmi_prop_SHADER_LOC "ShadingWithColor.fragmentshader");
		break;
	case RENDER_TEXTURE:
		programID = LoadShaders(nmi_prop_SHADER_LOC "ShadingWithTexture.vertexshader", nmi_prop_SHADER_LOC "ShadingWithTexture.fragmentshader");
		// Load the texture 
		Texture = loadBMP_custom(texture_path.c_str());
		// Get a handle for our "myTextureSampler" uniform
		TextureID = glGetUniformLocation(programID, "myTextureSampler");
		break;
	default:
		break;
	}

	// Get a handle for our "MVP" uniform
	MatrixID = glGetUniformLocation(programID, "MVP");

	// Projection matrix : 90° Field of View, 16:9 ratio, display range : 0.1 unit <-> 100 units
	//Projection = glm::perspective(glm::radians(75.0f), 16.0f / 9.0f, 0.1f, 100.0f);
	//Projection = glm::ortho(-16.0f,16.0f,-9.0f,9.0f, 1.0f, 100.0f);
	//---------------------------------------------------------

	double zn = near_clipping_plane; //nearest clipping plane
	double zf = far_clipping_plane; //faresst clipping plane

	Projection[0] = glm::vec4(fx / (-cx), 0.0f, 0.0f, 0.0f);
	Projection[1] = glm::vec4(0.0f, fy / (-cy), 0.0f, 0.0f);
	Projection[2] = glm::vec4(0.0f, 0.0f, (zn + zf) / (zn - zf), -1.0f);
	Projection[3] = glm::vec4(0.0f, 0.0f, 2 * zn*zf / (zn - zf), 0.0f);
	
	bool res;
	switch (RenderingMode)
	{
	case RENDER_POINT_CLOUD:
		// Read our .xyz file
		res = loadXYZ(cloud_path.c_str(), offset_path.c_str(), cloud_vertices, colors);
		//vertexbuffer
		glGenBuffers(1, &cloud_vertexbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, cloud_vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, cloud_vertices.size() * sizeof(glm::vec3), &cloud_vertices[0], GL_STATIC_DRAW);
		//colorbuffer
		glGenBuffers(1, &colorbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);
		break;
	case RENDER_TEXTURE:
		// Read our .obj file
		res = loadOBJ(object_path.c_str(), cloud_vertices, uvs);
		//vertexbuffer
		glGenBuffers(1, &cloud_vertexbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, cloud_vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, cloud_vertices.size() * sizeof(glm::vec3), &cloud_vertices[0], GL_STATIC_DRAW);
		//texcordsbuffer
		glGenBuffers(1, &uvbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);
		break;
	default:
		break;
	}
	return true;
}

//OpenGL initialization. Constructor uses it.
template <unsigned char RenderingMode>
bool Rendering<RenderingMode>::initGL()
{
	std::stringstream ss_log;
	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return false;
	}

	glfwWindowHint(GLFW_SAMPLES, 4); //4x MSAA 
	glfwWindowHint(GLFW_VISIBLE, 0); 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


	// Open a window and create its OpenGL context
	window = glfwCreateWindow(windowWidth, windowHeight, "Synthetic views", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		return false;
	}

	
	glfwMakeContextCurrent(window);

	// But on MacOS X with a retina screen it'll be 1024*2 and 768*2, so we get the actual framebuffer size:
	glfwGetFramebufferSize(window, &windowWidth, &windowHeight); 

	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return false;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	// Hide the mouse and enable unlimited mouvement
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Set the mouse at the center of the screen
	glfwPollEvents();
	glfwSetCursorPos(window, windowWidth / 2, windowHeight / 2);

	// Dark background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	glEnable(GL_MULTISAMPLE);

	switch (RenderingMode)
	{
	case RENDER_POINT_CLOUD:
		glPointSize(PointSize); //Set pointSize
		break;
	default:
		break;
	}

	//setting to tightly packed memory storage
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	const GLubyte* vendor = glGetString(GL_VENDOR); // Returns the vendor
	const GLubyte* renderer = glGetString(GL_RENDERER); // Returns a hint to the model
	int major = 0;
	int minor = 0;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	ss_log << vendor << " " << renderer << std::endl;
	ss_log << "OpenGL Version: " << major << "." << minor << std::endl;
	helperFunctions::log(ss_log, logPath);
	return true;
}

//Creates Frame buffers (in GPU memory). Constructor uses it.
template <unsigned char RenderingMode>
bool Rendering<RenderingMode>::CreateFrameBuffers()
{
	FramebufferName = 0;

	glGenFramebuffers(1, &FramebufferName);

	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

	// The texture we're going to render to
	glGenTextures(1, &renderedTexture);
	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, renderedTexture);

	// Give an empty image to OpenGL ( the last "0" means "empty" )

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageWidth, imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, 0);

	// Poor filtering
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

	// The depth buffer 
	glGenRenderbuffers(1, &depthrenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, imageWidth, imageHeight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return false;

	DrawBuffers[0] = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
	return true;
}

//Quad initializer to draw frame to screen. Constructor uses it.
template <unsigned char RenderingMode>
bool Rendering<RenderingMode>::initQuad()
{
	// The fullscreen quad's FBO
	GLfloat g_quad_vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f,  1.0f, 0.0f,
	};

	glGenBuffers(1, &quad_vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	// Create and compile our GLSL program from the shaders
	quad_programID = LoadShaders(nmi_prop_SHADER_LOC "TextureShader.vertexshader", nmi_prop_SHADER_LOC "TextureShader.fragmentshader");
	return true;
}

//Constructor
template <unsigned char RenderingMode>
Rendering<RenderingMode>::Rendering(const float& pointSize, const int&windowwidth, const int&windowheight, const int&imagewidth, const int&imageheight,
	const int &num_of_views_x, const int & num_of_views_y, const int & num_of_views_z, const float &stepx, const float & stepy, const float & stepz,
	const glm::vec3 &Cam_pos, const glm::vec3 &Cam_dir, const glm::vec3 &Cam_up, double near_clipping_plane, double far_clipping_plane, double fx, double fy, double cx, double cy, std::string object_path, std::string texture_path, std::string cloud_path, std::string offset_path, std::string log_Path) :
	PointSize(pointSize),
	windowWidth(windowwidth),
	windowHeight(windowheight),
	imageWidth(imagewidth),
	imageHeight(imageheight),
	numSynthX(num_of_views_x),
	numSynthY(num_of_views_y),
	numSynthZ(num_of_views_z),
	stepX(stepx),
	stepY(stepy),
	stepZ(stepz),
	Camera_pos(Cam_pos),
	Camera_direction(Cam_dir),
	Camera_up(Cam_up),
	current_level(num_of_views_z / 2),
	part_width(windowWidth / numSynthX),
	part_height(windowHeight / numSynthY),
	logPath(log_Path)
{
	initGL(); //initialise OpenGL
	initVBO(near_clipping_plane, far_clipping_plane,fx,fy,cx,cy, object_path, texture_path, cloud_path,  offset_path); //initialise VBO, load Model
	CreateFrameBuffers(); 
	initQuad(); 
}

//Renders one image
template <unsigned char RenderingMode>
bool Rendering<RenderingMode>::renderToTexture(const glm::vec3& translation, std::vector<unsigned char> &synth)
{
	
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
	glBindTexture(GL_TEXTURE_2D, renderedTexture);

	// Render to our framebuffer
	glViewport(0, 0, imageWidth, imageHeight); // Render on the whole framebuffer, complete from the lower left corner to the upper right

	// Clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	ViewMatrix = glm::lookAt(
		Camera_pos + translation, // Camera in World Space
		Camera_direction + translation, // and looks at the origin
		Camera_up// Head is up (set to 0,-1,0 to look upside-down)
	);
	
	MVP = Projection*ViewMatrix;// *ModelMatrix;

	glUseProgram(programID);

	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

	switch (RenderingMode)
	{
	case RENDER_POINT_CLOUD:
		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, cloud_vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride 
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : colors
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(
			1,                                // attribute
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride /
			(void*)0                          // array buffer offset
		);

		glDrawArrays(GL_POINTS, 0, cloud_vertices.size());	//this drawes the image
		break;
	case RENDER_TEXTURE:
		// Bind our texture in Texture Unit 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, Texture);
		// Set our "myTextureSampler" sampler to user Texture Unit 0
		glUniform1i(TextureID, 0);

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, cloud_vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute
			2,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		glDrawArrays(GL_TRIANGLES, 0, cloud_vertices.size());
		break;
	default:
		break;
	}

	glDisableVertexAttribArray(0);
 	glDisableVertexAttribArray(1);
	
	//reads the data from GPU memory to CPU memory                 
	glReadPixels(0, 0, imageWidth, imageHeight, GL_RED, GL_UNSIGNED_BYTE, &synth[0]);
	
	
	//unbind
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return true;
}

//Renders one image but leaves it on the GPU
template <unsigned char RenderingMode>
bool Rendering<RenderingMode>::renderToTextureOnGPU(const glm::vec3& translation)
{	
	///////////////SETUP as in renderSyntheticViews() /////////////////////
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
	glBindTexture(GL_TEXTURE_2D, renderedTexture);

	// Render to our framebuffer
	glViewport(0, 0, imageWidth, imageHeight); // Render on the whole framebuffer, complete from the lower left corner to the upper right

	// Clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	ViewMatrix = glm::lookAt(
		Camera_pos + translation, // Camera in World Space
		Camera_direction + translation, // and looks at the origin
		Camera_up// Head is up (set to 0,-1,0 to look upside-down)
	);

	MVP = Projection * ViewMatrix;// *ModelMatrix;

	glUseProgram(programID);

	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

	switch (RenderingMode)
	{
	case RENDER_POINT_CLOUD:
		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, cloud_vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride 
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : colors
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(
			1,                                // attribute
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride /
			(void*)0                          // array buffer offset
		);

		glDrawArrays(GL_POINTS, 0, cloud_vertices.size());	//this drawes the image
		break;
	case RENDER_TEXTURE:
		// Bind our texture in Texture Unit 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, Texture);
		// Set our "myTextureSampler" sampler to user Texture Unit 0
		glUniform1i(TextureID, 0);

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, cloud_vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute
			2,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		glDrawArrays(GL_TRIANGLES, 0, cloud_vertices.size());
		break;
	default:
		break;
	}

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);               
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return true;
}

//sets the camera coordinates (in global frame)
template <unsigned char RenderingMode>
bool Rendering<RenderingMode>::setCamera(const glm::vec3 &Cam_pos, const glm::vec3 &Cam_dir, const glm::vec3 &Cam_up)
{
	Camera_direction = Cam_dir;
	Camera_pos = Cam_pos;
	Camera_up = Cam_up;
	return true;
}

//calculates the transformations in the global frame, so we can shift the camera with these values
template <unsigned char RenderingMode>
glm::vec3 Rendering<RenderingMode>::calculateTranslation(int synthx, int synthy, int synthz)
{
	GLfloat x_offset = ((float)numSynthX - 1.0f) / 2.0f;
	GLfloat y_offset = ((float)numSynthY - 1.0f) / 2.0f;
	GLfloat z_offset = ((float)numSynthZ - 1.0f) / 2.0f;

	glm::vec3 translation;

	float length = sqrt(pow(Camera_up[0], 2) + pow(Camera_up[1], 2) + pow(Camera_up[2], 2));
	//cameraUp normalized
	glm::vec3 dir_y = glm::vec3((float)(Camera_up[0] / length), (float)(Camera_up[1] / length), (float)(Camera_up[2] / length));

	length = sqrt(pow(Camera_direction[0] - Camera_pos[0], 2) + pow(Camera_direction[1] - Camera_pos[1], 2) + pow(Camera_direction[2] - Camera_pos[2], 2));
	//InvViewDir normalized
	glm::vec3 dir_z = glm::vec3(-1.0*(float)((Camera_direction[0] - Camera_pos[0]) / length), -1.0*(float)((Camera_direction[1] - Camera_pos[1]) / length), -1.0*(float)((Camera_direction[2] - Camera_pos[2]) / length));

	//CameraDir normalized
	glm::vec3 dir_x = glm::rotate(dir_y, glm::radians(-90.0f), dir_z);

	translation = ((float)synthx - x_offset)*(float)stepX*dir_x + ((float)synthy - y_offset)*(float)stepY*dir_y + ((float)synthz - z_offset)*(float)stepZ*dir_z;
	return translation;
}

//calculates the transformations in the global frame, so we can shift the camera with these values (returns CV mat)
template < unsigned char RenderingMode>
cv::Mat Rendering<RenderingMode>::calculateTranslationCV(int synthx, int synthy, int synthz)
{
	GLfloat x_offset = ((float)numSynthX - 1.0f) / 2.0f;
	GLfloat y_offset = ((float)numSynthY - 1.0f) / 2.0f;
	GLfloat z_offset = ((float)numSynthZ - 1.0f) / 2.0f;

	glm::vec3 translation;

	float length = sqrt(pow(Camera_up[0], 2) + pow(Camera_up[1], 2) + pow(Camera_up[2], 2));
	//cameraUp normalized
	glm::vec3 dir_y = glm::vec3((float)(Camera_up[0] / length), (float)(Camera_up[1] / length), (float)(Camera_up[2] / length));

	length = sqrt(pow(Camera_direction[0] - Camera_pos[0], 2) + pow(Camera_direction[1] - Camera_pos[1], 2) + pow(Camera_direction[2] - Camera_pos[2], 2));
	//InvViewDir normalized
	glm::vec3 dir_z = glm::vec3(-1.0*(float)((Camera_direction[0] - Camera_pos[0]) / length), -1.0*(float)((Camera_direction[1] - Camera_pos[1]) / length), -1.0*(float)((Camera_direction[2] - Camera_pos[2]) / length));

	//CameraDir normalized
	glm::vec3 dir_x = glm::rotate(dir_y, glm::radians(-90.0f), dir_z);

	translation = ((float)synthx - x_offset)*(float)stepX*dir_x + ((float)synthy - y_offset)*(float)stepY*dir_y + ((float)synthz - z_offset)*(float)stepZ*dir_z;
	cv::Mat transl = cv::Mat(3, 1, CV_32F);
	transl.at<float>(0, 0) = translation.x;
	transl.at<float>(1, 0) = translation.y;
	transl.at<float>(2, 0) = translation.z;
	return transl;
}

//Destructor
template <unsigned char RenderingMode>
Rendering<RenderingMode>::~Rendering()
{
	// Cleanup VBO and shader
	glDeleteBuffers(1, &cloud_vertexbuffer);
	glDeleteBuffers(1, &colorbuffer);
	glDeleteBuffers(1, &quad_vertexbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteFramebuffers(1, &FramebufferName);
	glDeleteTextures(1, &renderedTexture);
	glDeleteRenderbuffers(1, &depthrenderbuffer);
}

template <unsigned char RenderingMode>
void Rendering<RenderingMode>::resizeKernel(const int &numsynthx, const int & numsynthy,
	const int & numsynthz, const float &stepx, const float & stepy, const float & stepz)
{
	setStep_x(stepx);
	setStep_y(stepy);
	setStep_z(stepz);

	setSynthetic_count_x(numsynthx);
	setSynthetic_count_y(numsynthy);
	setSynthetic_count_z(numsynthz);
}


template <unsigned char RenderingMode>
int Rendering<RenderingMode>::getImageWidth() { return imageWidth; }
template <unsigned char RenderingMode>
int Rendering<RenderingMode>::getImageHeight() { return imageHeight; }
template <unsigned char RenderingMode>
int Rendering<RenderingMode>::getNumSynthX() { return numSynthX; }
template <unsigned char RenderingMode>
int Rendering<RenderingMode>::getNumSynthY() { return numSynthY; }
template <unsigned char RenderingMode>
int Rendering<RenderingMode>::getNumSynthZ() { return numSynthZ; }
template <unsigned char RenderingMode>
float Rendering<RenderingMode>::getStepX() { return stepX; }
template <unsigned char RenderingMode>
float Rendering<RenderingMode>::getStepY() { return stepY; }
template <unsigned char RenderingMode>
float Rendering<RenderingMode>::getStepZ() { return stepZ; }

template <unsigned char RenderingMode>
unsigned int Rendering<RenderingMode>::getFramebufferName() { return FramebufferName; }
template <unsigned char RenderingMode>
unsigned int Rendering<RenderingMode>::getrenderedTexture() { return renderedTexture; }


template <unsigned char RenderingMode>
void Rendering<RenderingMode>::setSynthetic_count_x(int sx) { numSynthX = sx; }
template <unsigned char RenderingMode>
void Rendering< RenderingMode>::setSynthetic_count_y(int sy) { numSynthY = sy; }
template <unsigned char RenderingMode>
void Rendering<RenderingMode>::setSynthetic_count_z(int sz) { numSynthZ = sz; }
template <unsigned char RenderingMode>
void Rendering<RenderingMode>::setStep_x(float stx) { stepX = stx; }
template <unsigned char RenderingMode>
void Rendering<RenderingMode>::setStep_y(float sty) { stepY = sty; }
template <unsigned char RenderingMode>
void Rendering<RenderingMode>::setStep_z(float stz) { stepZ = stz; }