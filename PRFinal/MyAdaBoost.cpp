#include <core/core.hpp>
using namespace cv;
#include <math.h>
#include <iostream>
#include <string>
using namespace std;

#include "MyAdaBoost.h"

const char* WeightsFileName = "weights.w";

void MyAdaBoost::train(const Mat& samples, const Mat& labels, const int numWeak){
	CV_Assert( samples.rows == labels.rows );
	const int numSamples = samples.rows;
	const int numPosSamples = countNonZero(labels);
	const int numNegSamples = numSamples - numPosSamples;

	_alpha = Mat::zeros(numWeak, 1, CV_64FC1);
	_weights = Mat::ones(labels.size(), CV_64FC1);
	_weights /= 2.;
	for (int i = 0; i < numSamples; i++)
		_weights.at<double>(i) /= labels.at<double>(i) > 0 ? numPosSamples : numNegSamples;

	cout << "Start Training" << endl;
	for (int i = 0; i < numWeak; i++){
		normalizeWeight();
		cout << "Training Weak Classifier: " << i << endl;
		double err = getOptimalWeakClassifier(&_weaks, samples, labels, _weights);

		if (err >= .5){
			cout << "Total weak count is: " << i << endl;
			_alpha.resize(i);
			break;
		}
		if (err == 0.){
			cout << "Error equals zero" << endl;
			_alpha.resize(i);
			break;
		}

		_weaks.back().save(i);

		double beta = err / (1. - err);
		_weights *= pow(beta, 1 - err);
		_alpha.at<double>(i) = -log10(beta);
				
		saveAlpha();
	}
	saveAlpha();
	cout << "Training Process Terminated" << endl;
}

double MyAdaBoost::predict(const Mat& samples){
	double pred = 0., sum = 0.;
	for (int i = 0; i < _alpha.rows; i++){
		pred += _alpha.at<double>(i) * _weaks[i].predict(samples);
		sum += _alpha.at<double>(i);
	}
	return (pred - sum/2.);
}

void MyAdaBoost::normalizeWeight(){
	double sum = 0.;
	MatIterator_<double> it;
	for (it = _weights.begin<double>(); it != _weights.end<double>(); it++)
		sum += *it;

	_weights /= sum;
}

void MyAdaBoost::saveAlpha(){
	FileStorage fs(string(ModelDirectory) + WeightsFileName, FileStorage::WRITE);
	fs << "Alpha" << _alpha;
	fs.release();
}

void MyAdaBoost::load(){
	FileStorage fs(string(ModelDirectory) + WeightsFileName, FileStorage::READ);
	if (!fs.isOpened()){
		cout << "Cannot open weights file " << endl;
		return;
	}

	fs["Alpha"] >> _alpha;
	fs.release();
	for (int i = 0; i < _alpha.rows; i++){
		WeakClassifier w;
		w.load(i);
		_weaks.push_back(w);
	}
}