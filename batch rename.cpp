//#define LOAD_IMAGE_PATH ("D://video4_23.jpg") //這邊輸入圖檔的路徑
//#define SAVE_IMAGE_NAME ("video71_24sssss") //這邊會將圖片存在 資料夾images內，檔名為(SAVE_IMAGE_NAME).jpg

#include "faceDefine.h"
#include "FaceDetection.h"
#include "OpenCVASMAlignment.h"
#include "FaceCrop.h"
#include "string"
#include "vector"
#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <process.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>

using namespace EXP2013;
using namespace std;

vector<CvRect> faceWindow; //store each face's rect position
vector<CvMat*> alignmentResult; // stroe facial points

IplImage *drawimg;

FaceDetection *face_detector;
OpenCVASMAlignment *myAlignment;
FaceCrop *FaceCropper;
vector<FaceInfo> CropFaceSet;
vector<IplImage*> FaceSet;
vector<FP> FaceFP;

typedef struct {
	IplImage *img;
	char *filename;
}t;

IplImage* img_resize(IplImage* src_img, int new_width, int new_height)
{
	IplImage* des_img;
	des_img = cvCreateImage(cvSize(new_width, new_height), src_img->depth, src_img->nChannels);
	cvResize(src_img, des_img, CV_INTER_LINEAR);
	return des_img;
}

void RegThreadFun(void *param)
{
	t *args = (t*) param;
	double currentresult = -1;
	IplImage *img = (IplImage*) args->img;
	drawimg = cvCloneImage(img);

	faceWindow = face_detector->Detect(img);//人臉偵測
	if(faceWindow.size() != 0) //若有偵測到一個以上的人臉 已經改成只會保留偵測後最大的臉
	{
		char fn[100];
		drawimg = cvCloneImage(img);
		for(int i=0; i<faceWindow.size(); i++){
			cvRectangle(drawimg,cvPoint(faceWindow[i].x, faceWindow[i].y),cvPoint(faceWindow[i].x+faceWindow[i].width, faceWindow[i].y+faceWindow[i].height),CV_RGB(255,0,0));
			IplImage *cropping;
			cropping = cvCloneImage(img);
			cvSetImageROI(cropping, faceWindow[i]);
			cropping = img_resize(cropping, 48, 48);
			cvSaveImage(args->filename, cropping);
			cvReleaseImage(&cropping);
		}
		myAlignment->SetImage(img);
		for(int i=0; i<faceWindow.size(); i++)
			alignmentResult.push_back(myAlignment->calcAlignment(faceWindow[i]));
		for(int i=0; i<alignmentResult.size(); i++){
			for (int k = 0; k < alignmentResult[i]->rows; k++)
				if(k == 31 ||k == 36 || k == 66 || k == 67)
					cvCircle(drawimg, cvPoint(cvmGet(alignmentResult[i], k, 0), cvmGet(alignmentResult[i], k, 1)), 1, CV_RGB(255, 0, 0), 2);
				else
					cvCircle(drawimg, cvPoint(cvmGet(alignmentResult[i], k, 0), cvmGet(alignmentResult[i], k, 1)), 1, CV_RGB(0, 255, 0), 2);

			FP AlignOK;
			AlignOK.righteye.x = cvmGet(alignmentResult[i], 36, 0);
			AlignOK.righteye.y = cvmGet(alignmentResult[i], 36, 1);
			AlignOK.lefteye.x = cvmGet(alignmentResult[i], 31, 0);
			AlignOK.lefteye.y = cvmGet(alignmentResult[i], 31, 1);;
			AlignOK.mouth.x = cvmGet(alignmentResult[i], 66, 0);
			AlignOK.mouth.y = cvmGet(alignmentResult[i], 66, 1);
			AlignOK.nose.x = cvmGet(alignmentResult[i], 67, 0);
			AlignOK.nose.y = cvmGet(alignmentResult[i], 67, 1);
			FaceFP.push_back(AlignOK);
		}
		//若要將crop的位置及ASM 68個點的位置畫出 將以下註解拿掉
		/*
		sprintf(fn,"images/%s_drawing.jpg",args->filename);
		cvSaveImage(fn, drawimg);
		*/
	}// 若有找到人臉 則進行alignment
	else return;
}
void main(int argc, char *argv[])
{
	face_detector = new FaceDetection("cascade Data\\haarcascade.xml");
    myAlignment = new OpenCVASMAlignment();
	myAlignment->setModelPath("ASM Data/FrontalFace_best.amf");
	if (myAlignment->loadModel() == -1) {
		assert("Could not load ASM model...\n");
	}
	
	IplImage *img;
	IplImage *img1;
	char filename[100],savepath[100];
	int i = 0;
	char windowname[20];

	for (i = 1; i < 2726; i++){
		sprintf_s(filename, "D:\\Termproject-PR\\testFile\\%03d.jpg", i);
		sprintf_s(windowname, "Img%d", i);
		sprintf_s(savepath, "D:\\Termproject-PR\\testSave\\%03d.jpg", i);
		cvNamedWindow(windowname,CV_WINDOW_AUTOSIZE);
		img = cvLoadImage(filename);
		cout << filename << endl;
		if (!img) return;
		t* arg;
		arg = (t *)malloc(sizeof(t));
		arg->img = img;
		arg->filename = savepath;
		RegThreadFun(arg);
		cvShowImage(windowname,img);
		int cur_key = cvWaitKey(1000);
		cvReleaseImage(&img);
		cvDestroyWindow(windowname);
	}

	delete face_detector;
	delete myAlignment;
}