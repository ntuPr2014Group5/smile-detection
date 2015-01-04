// 
// Start of include openCV header
//


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;

//
// End of include openCv header
//

//
// Start of include C / C ++ header
//

#include <iostream>
#include <algorithm>
using namespace std;
using std::cout;
//
// End of include C / C ++ header
//


//
// Start of function prototype declaration
//

string int2str(int &i);

//
// End of function prototype declaration
//

string int2str(int &i)
{
	string s;
	stringstream ss(s);
	ss << i;
	return ss.str();
}//end of int2str


//int main(int argc, char* argv[])
//{
//	// Data for visual representation
//	int width = 512, height = 512;
//	Mat image = Mat::zeros(height, width, CV_8UC3);
//
//	// Set up training data
//	float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
//	Mat labelsMat(4, 1, CV_32FC1, labels);
//
//	float trainingData[4][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };
//	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
//
//	// Set up SVM's parameters
//	CvSVMParams params;
//	params.svm_type = CvSVM::C_SVC;
//	params.kernel_type = CvSVM::LINEAR;
//	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
//
//	// Train the SVM
//	CvSVM SVM;
//	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
//
//	Vec3b green(0, 255, 0), blue(255, 0, 0);
//	// Show the decision regions given by the SVM
//	for (int i = 0; i < image.rows; ++i)
//	for (int j = 0; j < image.cols; ++j)
//	{
//		Mat sampleMat = (Mat_<float>(1, 2) << j, i);
//		float response = SVM.predict(sampleMat);
//
//		if (response == 1)
//			image.at<Vec3b>(i, j) = green;
//		else if (response == -1)
//			image.at<Vec3b>(i, j) = blue;
//	}
//
//	// Show the training data
//	int thickness = -1;
//	int lineType = 8;
//	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
//	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
//	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
//	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
//
//	// Show support vectors
//	thickness = 2;
//	lineType = 8;
//	int c = SVM.get_support_vector_count();
//
//	for (int i = 0; i < c; ++i)
//	{
//		const float* v = SVM.get_support_vector(i);
//		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
//	}
//
//	imwrite("result.png", image);        // save the image
//
//	imshow("SVM Simple Example", image); // show it to the user
//	waitKey(0);
//
//}//end of main





//=======






//int main(int argc, char** argv)
//{
//	vector<int> compression_params;
//	compression_params.push_back(100);
//	const int input_max_num = 2174;
//
//	for (int i = 1; i <= input_max_num; i++){
//
//		Mat image;
//		Mat imgResize;
//
//		string idx = int2str(i);
//		//string location = "Smile48\\" + idx + ".jpg";
//		//string outputName = "Smile48gray\\" + idx + ".jpg";
//
//		string location = "NonSmile48\\" + idx + ".jpg";
//		string outputName = "NonSmile48gray\\" + idx + ".jpg";
//
//		image = imread(location, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
//
//		if (!image.data)                              // Check for invalid input
//		{
//			cout << "Could not open or find the image" << std::endl;
//			return -1;
//		}
//
//		namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
//		//imshow("Display window", image);                   // Show our image inside it.
//		//cout << image.channels() << endl;
//
//		imgResize.push_back(image.reshape(0, 1));
//		//imshow("Display window", imgResize);                   // Show our image inside it.
//
//		imwrite(outputName, imgResize, compression_params);  // loseless image save
//
//
//		//waitKey(0);                                          // Wait for a keystroke in the window
//
//
//	}
//
//	return EXIT_SUCCESS;
//}


// ============


int main(int argc, char** argv)
{
	// SVM 訓練分類器
	CvSVM SVM;

	// 訓練樣本數目的總張數
	int trainImgCount = 4000;
	int img_area = 48 * 48;

	// 是否讀取SVM train model 檔案的 boolean
	bool isLoadSvmModel = true;

	if (isLoadSvmModel == false )
	{

		vector<int> compression_params;
		compression_params.push_back(100);

	
		const int input_max_size = 50;

		Mat samples;
		Mat vectorImg;
		const int vectorLength = 48 * 48;

		string outputName = "allSampleVectorImage.jpg";

		//samples.resize( 4000, vectorLength );



		Mat training_mat(trainImgCount, img_area, CV_32FC1);

		// for smile face
		for (int file_num = 1; file_num <= trainImgCount / 2; file_num++)
		{
			string idx = int2str( file_num );
			string inputName = "Smile48gray\\" + idx + ".jpg";

			Mat img_mat = imread(inputName, 0); // I used 0 for greyscale
			int ii = 0; // Current column in training_mat
			for (int i = 0; i < img_mat.rows; i++) {
				for (int j = 0; j < img_mat.cols; j++) {
					training_mat.at<float>( file_num-1 , ii++) = img_mat.at<uchar>(i, j);
				}
			}

		}//end of for loop


		// for non-smile face
		for (int file_num = 1; file_num <= trainImgCount / 2; file_num++)
		{
			string idx = int2str(file_num);
			string inputName = "NonSmile48gray\\" + idx + ".jpg";

			Mat img_mat = imread(inputName, 0); // I used 0 for greyscale
			int ii = 0; // Current column in training_mat
			for (int i = 0; i < img_mat.rows; i++) {
				for (int j = 0; j < img_mat.cols; j++) {
					training_mat.at<float>(trainImgCount / 2 + file_num - 1, ii++) = img_mat.at<uchar>(i, j);
				}
			}

		}//end of for loop




		//namedWindow("Display window", WINDOW_AUTOSIZE ); // Create a window for display.
		//imshow("Display window", samples );              // Show our image inside it.
		//imwrite(outputName, samples, compression_params);  // loseless image save
	
		imwrite(outputName, training_mat, compression_params);  // loseless image save



		//
		//
		// ===================
		//
		// Start of SVM training
		//
		//



		// Data for visual representation
		//int width = 512, height = 512;
		//Mat image = Mat::zeros(height, width, CV_8UC3);

		// Set up training data
		//float labels[4] = { 1.0, -1.0, -1.0, -1.0 };

		// 前50張 image 是笑臉 , label value = 1;
		// 後50張 image 是非笑臉, label value = 0.0
		//float labels[100] = {	1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		//						1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
		//						1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		//						1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		//						1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		//						0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		//						0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		//						0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		//						0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		//						0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		//};
		float labels[4000] = { 0 };
		// 前2000張 是 笑臉
		fill(labels, labels + trainImgCount/2, 1);
		// 把float型別的label array 灌入 labelsMat
		Mat labelsMat(trainImgCount, 1, CV_32FC1, labels);


		//const int vectorLength = 2304; // 48 * 48 = 2304 = row vector length per image with w = 48, h = 48

		//float trainingData[4][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };
		//Mat trainingDataMat(100, vectorLength, CV_32FC1, samples);
		//Mat trainingDataMat(4, 2, CV_32FC1, samples);
		//Mat trainingDataMat(Size(vectorLength, 100), CV_32FC1);
		//samples.copyTo(trainingDataMat);

		// Set up SVM's parameters
		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		// Train the SVM
		//CvSVM SVM;
		//training_mat
		//SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
		SVM.train(training_mat, labelsMat, Mat(), Mat(), params);

		cout << "Train Successul" << endl;


		//
		//
		// ===== code seperation line ====
		//
		//


		//SVM.save("svmRawModel.mdl");

		std::cout << "SVM Raw Model Save Successful" << endl;
	
	}
	// load SVM model
	
	SVM.load("svmRawModel.mdl");

	cout << "SVM Raw Model Load Successful" << endl;

	int testing_file_num = 2100;
	string idx = int2str(testing_file_num);
	string inputName = "Smile48gray\\" + idx + ".jpg";

	int testImgCount = 1;
	Mat img_Test(testImgCount, img_area, CV_32FC1);
	Mat img_test_input = imread(inputName, 0); // I used 0 for greyscale

	int ii = 0;
	for (int i = 0; i < img_test_input.rows; i++) {
		for (int j = 0; j < img_test_input.cols; j++) {
			img_Test.at<float>(0, ii++) = img_test_input.at<uchar>(i, j);
		}
	}

	cout << SVM.predict(img_Test);


	//svm.predict(img_mat_1d);

	//waitKey(0);
	return EXIT_SUCCESS;
}//end of main