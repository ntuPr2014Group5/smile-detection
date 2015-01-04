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







//=======




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
			const int vectorLength = (24*24);
			///const int vectorLength = ( 48 * 48 ) *( 48 * 48 - 1 ) / 2;
			string outputName = "allSampleVectorImage.jpg";

			//samples.resize( 4000, vectorLength );


			// Raw profile
			//Mat training_mat(trainImgCount, img_area, CV_32FC1);

			// Pixel difference
			Mat training_mat(trainImgCount, vectorLength, CV_32FC1);
			/*
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

			*/

			//
			//
			// ===== Code Separation Line
			//
			//


			
			// for smile face with pixel difference
			for (int file_num = 1; file_num <= trainImgCount / 2; file_num++)
			{
				string idx = int2str(file_num);
				string inputName = "Smile48\\" + idx + ".jpg";
				Mat img_mat = imread(inputName, 0);				// I used 0 for greyscale

				// load image and extract feature
				int featurePos = 0;
				for (int i = 24; i < img_mat.rows; i++) {

					for (int j = 12; j < 36; j++) {
						float center = img_mat.at<uchar>(36, 36);
						float cursor = img_mat.at<uchar>(i, j);
						float pxlDifference = center - cursor;
						training_mat.at<float>( file_num - 1, featurePos++) = pxlDifference;
					
					}//end of for j
				}//end of for i

			}//end of for loop

			//system("pause");

			// for non smile face with pixel difference
			for (int file_num = 1; file_num <= trainImgCount / 2; file_num++)
			{
				string idx = int2str(file_num);
				string inputName = "NonSmile48\\" + idx + ".jpg";
				Mat img_mat = imread(inputName, 0);				// I used 0 for greyscale

				// load image and extract feature
				int featurePos = 0;
				for (int i = 24; i < img_mat.rows; i++) {

					for (int j = 12; j < 36; j++) {
						float center = img_mat.at<uchar>(36, 36);
						float cursor = img_mat.at<uchar>(i, j);
						float pxlDifference = center - cursor;
						training_mat.at<float>(trainImgCount / 2 + file_num - 1, featurePos++) = pxlDifference;
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
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 400, 1e-9);

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


			SVM.save("svmRawModel.mdl");

			std::cout << "SVM Raw Model Save Successful" << endl;
	
	}//end of if isLoadSvmModel
	else
	{

			// load SVM model
	
			SVM.load("svmRawModel.mdl");

			cout << "SVM Raw Model Load Successful" << endl;

	}//end of if isLoadSvmModel else...



	for (int testing_file_num = 1; testing_file_num <= 98; testing_file_num++)
	{
	
		//  在 testing_file_num 放上要測試的圖片編號
		//int testing_file_num = 2001;
		//string idx = int2str(5);
		string idx = int2str(testing_file_num);
		// 在 inputName指定要測試笑臉的圖案 還是 非笑臉的圖案
		//string inputName = "Smile48\\" + idx + ".jpg";
		//string inputName = "NonSmile48\\" + idx + ".jpg";
		string inputName = "labData\\" + idx + ".jpg";

		int testImgCount = 1;
		Mat img_Test(testImgCount, 24*24, CV_32FC1);
		Mat img_test_input = imread(inputName, 0); // I used 0 for greyscale

		imshow("show", img_test_input);

		int ii = 0;
		for (int i = 24; i < img_test_input.rows; i++) {
			for (int j = 12; j < 36; j++) {
				float center = img_test_input.at<uchar>(36, 36);
				float cursor = img_test_input.at<uchar>(i, j);
				float pxlDifference = center - cursor;
				
				img_Test.at<float>(0, ii++) = pxlDifference;
			}
		}

		cout << "img id : " << testing_file_num << " : " << SVM.predict(img_Test) << endl;

		//system("pause");

	}//end of for loop testing_file_num 
	
	waitKey(0);
	return EXIT_SUCCESS;
}//end of main











