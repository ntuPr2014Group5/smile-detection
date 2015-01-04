#define _CRT_SECURE_NO_WARNINGS

#include "MyAdaBoost.h"

#include <windows.h>
#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
string int2str(int &i);

int main(int argc, char** argv)
{
	vector<int> compression_params;
	compression_params.push_back(100);

	const int input_max_size = 100;

	Mat samples;
	Mat vectorImg;
	const int vectorLength = 48 * 48;

	string outputName = "sampleVectorImage.jpg";

	//samples.resize( 4000, vectorLength );

	for (int i = 1; i <= input_max_size; i++)
	{
		Mat image;
		string idx = int2str(i);
		string inputName = "Smile48gray\\" + idx + ".jpg";
		//string inputName = "NonSmile48gray\\" + idx + ".jpg";
		vectorImg = imread(inputName, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
		samples.push_back( vectorImg );
	}//end of loop

	for (int i = 1; i <= input_max_size; i++)
	{
		Mat image;
		string idx = int2str(i);
		//string inputName = "Smile48gray\\" + idx + ".jpg";
		string inputName = "NonSmile48gray\\" + idx + ".jpg";
		vectorImg = imread(inputName, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
		samples.push_back(vectorImg);
	}//end of loop

	//namedWindow("Display window", WINDOW_AUTOSIZE );// Create a window for display.
	//imshow("Display window", samples );                   // Show our image inside it.
	imwrite(outputName, samples, compression_params);  // loseless image save


	//
	//
	// ===================
	Mat label(2*input_max_size, 1, 6);

	for (int i = 0; i < input_max_size; i++)
	{
		label.at<double>(i, 0) = 1;
	}//end of loop

	for (int i = input_max_size; i < 2*input_max_size; i++)
	{
		label.at<double>(i, 0) = 0;
	}//end of loop


	MyAdaBoost trainMachine;

	trainMachine.train(samples, label, 500);

	//waitKey(0);
	return EXIT_SUCCESS;
}

string int2str(int &i)
{
	string s;
	stringstream ss(s);
	ss << i;
	return ss.str();
}