#include <iostream>
using namespace std;
#include <imgproc/imgproc.hpp>
#include <objdetect/objdetect.hpp>

#include "MyClassifier.h"

#define SVMFileName "SVMModel\\HOG_SVMParams.yml"

const int SVM_IMG_WIDTH = 48;
const int SVM_IMG_HEIGHT = 48;

MyClassifier::MyClassifier(){
	//_knuckle_svm.load(Knuckle_SVMFileName);
}

double
MyClassifier::classify(const Mat& image){
	return _svm.predict(getHOGFromGray(image));
}

void
MyClassifier::train_auto(const Mat& trainData, const Mat& labels){
	
	// prepare svm parameter for auto-train
	SVMParams params;
	params.svm_type = SVM::C_SVC;			// C-SVM
	params.kernel_type = SVM::LINEAR;		// RBF kernel
	params.gamma = 0;
	bool balanced = true;

	Mat td, lb;
	getHOGFromGray(trainData).convertTo(td, CV_32F);
	labels.convertTo(lb, CV_32F);

	// train knuckle svm
	cout << "Starting training svm" << endl;
	_svm.train_auto(td, lb, Mat(), Mat(), params, 4,
		SVM::get_default_grid(SVM::C), SVM::get_default_grid(SVM::GAMMA), SVM::get_default_grid(SVM::P),
		SVM::get_default_grid(SVM::NU), SVM::get_default_grid(SVM::COEF), SVM::get_default_grid(SVM::DEGREE), balanced);

	cout << "Finished training svm" << endl;

	// save svm parameters
	_svm.save(SVMFileName);
}

void
MyClassifier::load(){
	_svm.load(SVMFileName);	
}

// assume features are fed row by row
Mat
MyClassifier::getHOGFromGray(const Mat& input){
	CV_Assert(input.type() == CV_8UC1);

	Mat output;

	for (int i = 0; i < input.size().height; i++){
		Mat im = input.row(i).clone();
		im = im.reshape(1, SVM_IMG_HEIGHT);	// reshape to CV_8UC3
		// --------------------preprocessing--------------------
		equalizeHist(im, im);				// equalize histogram
		// -----------------------------------------------------

		// Hog feature
		Size hogWindowSize(SVM_IMG_WIDTH, SVM_IMG_HEIGHT);
		Size hogBlockSize(16, 16);
		Size hogBlockStride(8, 8);
		Size hogCellSize(8, 8);
		int hogNBins = 9;
		HOGDescriptor hog(hogWindowSize, hogBlockSize, hogBlockStride, hogCellSize, hogNBins);
		vector<float> descriptorValues;
		hog.compute(im, descriptorValues);

		Mat myFeature(descriptorValues);
		myFeature = myFeature.reshape(0, 1);// fold to row vector
		output.push_back(myFeature);
	}
	return output;
}
