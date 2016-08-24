#include "stdafx.h"

#include "vl/dsift.h"
#include "vl/sift.h"
#include <opencv2/opencv.hpp>
#include "highgui.h"
#include "cv.h"
#include <string>
#include <fstream>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

extern void GetFileList( const string& directory, vector<string>* filelist );
extern VlDsiftFilter * initialize_vldsift_filter(int width, int height, int stepX, int stepY);
extern void gzh_vlfeat_sift_llc_max_pooling(Mat &src, Mat &vol, VlDsiftFilter *filter, 
									 const VlDsiftKeypoint *keyPoints, Mat &des,
									 Mat &descriptor_after_pooling, int knn, int pLevels);


//needed variable
//for filter
Mat des;
const VlDsiftKeypoint *keyPoints;
VlDsiftFilter *filter;
//for vocabulary
const string vocabularyFilePath("vlfeat_svm_result/vocabulary.xml.gz");
Mat vocabulary;
//for svm
const string svmFilePath("vlfeat_svm_result/dense_sift_svms/palm+victor.xml");
CvSVM  *psvm;

bool prepare_for_feature_retrieve()
{
	//This function is used to prepare the <VlDsiftFilter> <vocabulary> and <CvSVM>
	//Step 1: the <VlDsiftFilter *>
	//在实际运行过程中，将图像块缩放为60*60进行处理，因此对于
	//所有待处理的图像来说，尺寸最终都是60*60，因此可以使用同一个
	//<VlDsiftFilter *> 
	int sample_siz = 60;
	//build the filter and allocate the necessary space
	int stepX = 6;
	int stepY = 6;
	int sift_dim = 128;
	filter = initialize_vldsift_filter(sample_siz, sample_siz, stepX, stepY);
	//可以将<const VlDsiftKeypoint *keyPoints> <Mat des>做成全局变量，那么避免了
	//初始化时不知道分配多少空间的尴尬了
	//allocate the space
	//num is the number of the points in this image
	int num = vl_dsift_get_keypoint_num(filter);
	//const VlDsiftKeypoint *
	keyPoints = new VlDsiftKeypoint[num];
	//Mat des;
	des.create(num, sift_dim, CV_32FC1);
	//Step 2:<vocabulary> 
	FileStorage fs( vocabularyFilePath, FileStorage::READ );
	if ( fs.isOpened() ) {
		fs["vocabulary"] >> vocabulary;
	}else{
		cout<<"vocabulary can not be loaded successfully"<<endl;
		return false;
	}
	//Step 3: <CvSVM>
	psvm = new CvSVM;
	psvm->load( svmFilePath.c_str() );
	return true;
}

float vlfeat_dsift_svm_hand_classifying( Mat& image)
{
	//In this function, these variables are needed here but defined in the whole range of this project.
	/*
	Mat des;
	const VlDsiftKeypoint *keyPoints;
	VlDsiftFilter *filter;
	Mat vocabulary
	CvSVM* psvm
	*/
	//Step 1: extract the feature
	//This is the process to get the feature of the image.
	int pLevels = 2;
	int knn = 5;
	Mat queryDescriptor;
	gzh_vlfeat_sift_llc_max_pooling(image, vocabulary, filter, keyPoints, des, queryDescriptor, knn, pLevels);
	//Step 2: using svm to classify
	float response =psvm->predict(queryDescriptor);
	//Here, if response = -1, it means the palm, but if it is 1, it means victory. 
	//printf("****%f****\n",response);
	return response;
}

int main_test_vlfeat_related()
{
	prepare_for_feature_retrieve();
	getchar();
	return 0;
}