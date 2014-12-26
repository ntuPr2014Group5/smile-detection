#include <vector>
using namespace std;

#include <core/core.hpp>
using namespace cv;

#define WIDTH 48
#define HEIGHT 48
#define DIM (WIDTH*HEIGHT)

const int NumTrainData = 100;

// assume input features are gray images folded into row vectors
Mat trainData(100, DIM, CV_8UC1);
Mat labels(100, 1, CV_8UC1);

class WeakClassifier{
	public:
		WeakClassifier(Point p1, Point p2){
			_theta = 0;
			_parity = 1;
			_pixels[0] = p1;
			_pixels[1] = p2;
		}

		double train(const Mat& trainData, const Mat& labels, const Mat& weights){
			CV_Assert(trainData.cols == DIM && trainData.type() == CV_8UC1);
			int* d = new int[trainData.rows];
			int* w = new int[trainData.rows];
			for (int i = 0; i < trainData.rows; i++){
				d[i] = diff(trainData.row(i));
				w[i] = weights.at<float>(i);
			}

			double opt_error = 0.5;

			for (int theta = 0; theta <= 255; theta++){
				double error = 0.;
				int parity = 1;
				for (int i = 0; i < trainData.rows; i++){
					if (parity*(d[i] - theta) < 0)
						error += w[i];
				}

				if (error > 0.5){
					parity *= -1;
					error = 1 - error;
				}

				if (error < opt_error){
					opt_error = error;
					_theta = theta;
					_parity = parity;
				}
			}
			return opt_error;
		}
		double predict(const Mat& input){
			return (_parity*diff(input) < _parity*_theta) ? 1. : 0.;
		}
	private:
		Point _pixels[2];
		int _theta, _parity;

		int diff(const Mat& input){
			int offset1 = _pixels[0].y*WIDTH + _pixels[0].x;
			int offset2 = _pixels[1].y*WIDTH + _pixels[1].x;
			return input.at<uchar>(offset1) - input.at<uchar>(offset2);
		}
};

double getOptimalWeakClassifier(vector<WeakClassifier>* weakVector, const Mat& weights){
	double error = 1.;
	

	return error;
}