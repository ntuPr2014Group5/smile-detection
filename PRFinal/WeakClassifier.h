#ifndef _WEAKCLASSIFIER_H_
#define _WEAKCLASSIFIER_H_

#include <vector>
#include <string>
using namespace std;

#include <core/core.hpp>
using namespace cv;

#define WIDTH 48
#define HEIGHT 48
#define DIM (WIDTH*HEIGHT)
#define OFFSET(x,y) (y*WIDTH + x)
#define TOPOINT(x) (Point(x%WIDTH, x/WIDTH))

const int NumTrainData = 100;
const char* ModelDirectory = "Models\\";

// assume input features are gray images folded into row vectors
Mat trainData(100, DIM, CV_8UC1);
Mat labels(100, 1, CV_8UC1);


class WeakClassifier{
	public:
		WeakClassifier(){
			_theta = 0;
			_parity = 1;
		}

		void resetTo(Point p1, Point p2){
			_theta = 0;
			_parity = 1;
			_pixels[0] = p1;
			_pixels[1] = p2;
		}

		const WeakClassifier& operator=(const WeakClassifier& w){
			_theta = w._theta;
			_parity = w._parity;
			_pixels[0] = w._pixels[0];
			_pixels[1] = w._pixels[1];
			return *this;
		}

		double train(const Mat& trainData, const Mat& labels, const Mat& weights){
			int* d = new int[trainData.rows];
			int* w = new int[trainData.rows];
			for (int i = 0; i < trainData.rows; i++){
				d[i] = diff(trainData.row(i));
				w[i] = weights.at<double>(i);
			}

			double opt_error = 0.5;

			for (int theta = 0; theta <= 255; theta++){
				double err = 0.;
				int parity = 1;
				for (int i = 0; i < trainData.rows; i++){
					if (d[i] - theta < 0)
						err += w[i];
				}

				if (err > 0.5){
					parity *= -1;
					err = 1. - err;
				}

				if (err < opt_error){
					opt_error = err;
					_theta = theta;
					_parity = parity;
				}
			}
			return opt_error;
		}

		double predict(const Mat& input){
			return (_parity*(diff(input) - _theta) < 0) ? 1. : 0.;
		}

		void save(int index){
			FileStorage fs(ModelDirectory + to_string(index) + ".mod", FileStorage::WRITE);
			fs << "Model" << "[";
			fs << "{:" << "p0_x" << _pixels[0].x << "p0_y" << _pixels[0].y
				<< "p1_x" << _pixels[1].x << "p1_y" << _pixels[1].y
				<< "theta" << _theta << "parity" << _parity
				<< "}";
			fs.release();
		}

		void load(int index){
			FileStorage fs(ModelDirectory + to_string(index) + ".mod", FileStorage::READ);
			if(!fs.isOpened())
				return;

			FileNode model = fs["Model"];
			_pixels[0] = Point(model["p0_x"], model["p0_y"]);
			_pixels[1] = Point(model["p1_x"], model["p1_y"]);
			_theta = (int)model["theta"];
			_parity = (int)model["parity"];
			fs.release();
		}

		friend bool isExisted(vector<WeakClassifier>*, const int, const int);
	private:
		Point _pixels[2];
		int _theta, _parity;

		inline int diff(const Mat& input){
			return input.at<uchar>(OFFSET(_pixels[0].x, _pixels[0].y)) - input.at<uchar>(OFFSET(_pixels[1].x, _pixels[1].y));
		}


};

bool isExisted(vector<WeakClassifier>* weakVector, const int i, const int j){
	for (int idx = 0; idx < weakVector->size(); idx++){
		WeakClassifier w = (*weakVector)[idx];
		int of1 = OFFSET(w._pixels[0].x, w._pixels[0].y);
		int of2 = OFFSET(w._pixels[1].x, w._pixels[1].y);
		if ( (of1 == i && of2 == j) || (of1 == j && of2 == i) )
			return true;
	}
	return false;
}

double getOptimalWeakClassifier(vector<WeakClassifier>* weakVector, const Mat& trainData, const Mat& labels, const Mat& weights){
	CV_Assert(trainData.cols == DIM && trainData.type() == CV_8UC1);
	CV_Assert(trainData.rows == labels.rows && labels.rows == weights.rows);

	double opt_error = 0.5;
	WeakClassifier opt_weak, weak;
	for (int i = 0; i < trainData.cols; i++){
		for (int j = 0; j < i; j++){
			if (isExisted(weakVector, i, j))
				continue;

			weak.resetTo(TOPOINT(i), TOPOINT(j));
			double err = weak.train(trainData, labels, weights);

			if (err < opt_error){
				opt_error = err;
				opt_weak = weak;
			}
		}
	}

	if (opt_error < 0.5){
		weakVector->push_back(opt_weak);
	}
	return opt_error;
}

#endif