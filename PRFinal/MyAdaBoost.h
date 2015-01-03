#ifndef _MYADABOOST_H_
#define _MYADABOOST_H_

#include <vector>
using namespace std;
#include "WeakClassifier.h"

class MyAdaBoost{
	public:
		MyAdaBoost(){}
		void train(const Mat&, const Mat&, const int = 500);
		double predict(const Mat&);		
		void load();
	private:
		vector<WeakClassifier> _weaks;
		Mat _weights;
		Mat _alpha;

		void saveAlpha();
		void normalizeWeight();
};


#endif