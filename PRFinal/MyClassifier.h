#ifndef _MYCLASSIFIER_H_
#define _MYCLASSIFIER_H_

#include <core/core.hpp>
#include <ml/ml.hpp>
using namespace cv;

class MyClassifier{
	public:
		MyClassifier();
		void train_auto(const Mat&, const Mat&);
		double classify(const Mat&);
		void load();

	private:
		CvSVM _svm;
		
		Mat getHOGFromGray(const Mat&);
};

#endif