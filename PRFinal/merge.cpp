#define _CRT_SECURE_NO_WARNINGS

#include "MyAdaBoost.h"
#include "MyClassifier.h"

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
void train();
void verify();
void demo();

int main(int argc, char** argv)
{
	//verify();
	demo();
	return EXIT_SUCCESS;
}

string int2str(int &i)
{
	string s;
	stringstream ss(s);
	ss << i;
	return ss.str();
}

void train(){
	const int input_max_size = 2000;

	Mat samples;
	Mat vectorImg;

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


	//MyAdaBoost trainMachine;
	//trainMachine.train(samples, label, 500);

	MyClassifier svm;
	svm.train_auto(samples, label);
}

void demo(){
	MyClassifier svm;
	svm.load();

	for(int i = 1; i <= 100; i++, i++){
		string inputName = int2str(i) + ".jpg";		
		Mat image = imread(inputName, CV_LOAD_IMAGE_GRAYSCALE);	

		if (!image.data)
			return;

		image = image.reshape(0, 1);

		cout << svm.classify(image) << endl;
	}
}

void verify(){
	int idx = 2001;
	MyClassifier svm;
	svm.load();

	for(int i = 0; i < 59; i++, idx++){
		string inputName = "Smile48gray\\" + int2str(idx) + ".jpg";
		Mat image = imread(inputName, CV_LOAD_IMAGE_GRAYSCALE);	

		cout << svm.classify(image) << endl;
		image = image.reshape(1, 48);
		imshow("hi", image);
		cvWaitKey(0);

		inputName = "NonSmile48gray\\" + int2str(idx) + ".jpg";
		image = imread(inputName, CV_LOAD_IMAGE_GRAYSCALE);

		cout << svm.classify(image) << endl;
		image = image.reshape(1, 48);
		imshow("hi", image);
		cvWaitKey(0);
	}
}