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
using std::cout;
string int2str(int &i);




#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	vector<int> compression_params;
	compression_params.push_back(100);


	const int input_max_size = 2000;

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
	//
	//




	//Mat label(1, 4000, 6);

	Mat label(4000, 1, 6);

	//label = Mat::zeros(3, 3, 6);
	//label.resize(1, 4000);
	cout << label.size() << endl;

	for (int i = 0; i < 2000; i++)
	{
		label.at<double>(i, 0) = 1;
	}//end of loop

	for (int i = 2000; i < 4000; i++)
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