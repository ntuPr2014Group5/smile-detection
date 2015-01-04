#ifndef _WEAKCLASSIFIER_H_
#define _WEAKCLASSIFIER_H_

#include <core/core.hpp>
using namespace cv;

extern const char* ModelDirectory;

class WeakClassifier{
	public:
		WeakClassifier();

		void resetTo(Point p1, Point p2);

		const WeakClassifier& operator=(const WeakClassifier& w);

		double train(const Mat& trainData, const Mat& labels, const Mat& weights);

		double train_gpu();

		double predict(const Mat& input);

		void save(int index);

		void load(int index);

		friend bool isExisted(vector<WeakClassifier>*, const int, const int);
	private:
		Point _pixels[2];
		int _theta, _parity;

		int diff(const Mat& input);
};

extern bool isExisted(vector<WeakClassifier>* weakVector, const int i, const int j);

extern double getOptimalWeakClassifier(vector<WeakClassifier>* weakVector, const Mat& trainData, const Mat& labels, const Mat& weights);

#endif