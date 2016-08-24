#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <memory>
#include <functional>
#include <queue>
#include <map>
#include <cstring>
#include <stdio.h>
#include  <afx.h>
#include "direct.h"

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

//LibLinear
#include "linear.h"

using namespace cv;
using namespace std;

#define LIBLINEAR 0
#define MAX_PATH 512

#define MIN_KPS    3
#define VOCA_COLS  3000
#define ClASS 3
#define esp 1e-10
#define TIME 0
#define TIME_SVM 1

#define DENSE_SIFT 1
#define FEATURE_SCALE 16.0f
#define TIME_SIFT_LLC 1

#define USE_VLFEAT 1

#include "vl/dsift.h"
#include "vl/sift.h"
#include "highgui.h"
#include "cv.h"
#include <string>
#include <fstream>
#include <iostream>

extern void getTime(string file, double time);
extern void getMatToFile(string file, Mat data);
extern Mat gzh_vlfeat_sift(Mat &src);
extern Mat getImageDescriptor(Mat &src, Mat &vol, int pLevles);
extern void gzh_vlfeat_sift_llc_max_pooling(Mat &src, Mat &vol, VlDsiftFilter *filter, 
									 const VlDsiftKeypoint *keyPoints, Mat &des,
									 Mat &descriptor_after_pooling, int knn, int pLevels);
extern VlDsiftFilter * initialize_vldsift_filter(int width, int height, int stepX, int stepY);
extern void vlfeatSIFT_points_extract_same_filter(Mat &src, VlDsiftFilter *dSIFT_filter);

extern Mat des;
extern const VlDsiftKeypoint *keyPoints;
extern VlDsiftFilter *filter;

void MakeDir( const string& filepath );
void help( const char* progName );
void GetDirList( const string& directory, vector<string>* dirlist );
void GetFileList( const string& directory, vector<string>* filelist );

const string kVocabularyFile( "vocabulary.xml.gz" );
#if !DENSE_SIFT
	const string kBowImageDescriptorsDir( "bagOfWords" );
	const string kSvmsDirs( "svms" );
#else
	const string kBowImageDescriptorsDir( "dense_sift_bagOfWords" );
	const string kSvmsDirs( "dense_sift_svms" );
#endif

class Params {
public:
#if !DENSE_SIFT
	Params(): wordCount( VOCA_COLS ), detectorType( "SIFT" ),
		descriptorType( "SIFT" ), matcherType( "FlannBased" ){ }
#else
	Params(): wordCount( VOCA_COLS ), detectorType( "Dense" ),
		descriptorType( "SIFT" ), matcherType( "FlannBased" ){ }
#endif
	int                wordCount;
	string        detectorType;
	string        descriptorType;
	string        matcherType;
};

#define BASE_BUFF_MAX_LEN 10
#define VocabularyPath "..\\vocabulary\\vocabulary1800_mysample.txt"

void getMatPrint(Mat &queryDescriptor)
{
	fstream f;
		f.open("queryDescriptor.txt", ios::out);
		if(f.is_open()){
			f.clear();
			f<<queryDescriptor;
			f.close();
		}
}

Mat GetVocabulary()
{

	int bufLen = BASE_BUFF_MAX_LEN;

	char *buf = new char[bufLen];

	ifstream fin;

	fin.open(VocabularyPath, ios::in);

	Mat src=Mat(VOCA_COLS, 128, CV_64FC1);
	string s;
	double num;

	while(getline(fin,s))
	{

		int j = 0;
		istringstream stream(s);
		for(int i=0; i<VOCA_COLS; i++){
			stream>>num;
			src.at<double>(i,j) = num;
		}
		j++;
	}
	return src;
}


/*
* loop through every directory
* compute each image's keypoints and descriptors
* train a vocabulary
*/
Mat BuildVocabulary( const string& databaseDir,
					const vector<string>& categories,
					const Ptr<FeatureDetector>& detector,
					const Ptr<DescriptorExtractor>& extractor,
					int wordCount)
{
	Mat allDescriptors;
	for ( int index = 0; index != categories.size(); ++index ) {
		cout << "processing category " << categories[index] << endl;
		string currentCategory = databaseDir +'/' + categories[index] + '/';
		vector<string> filelist;
		GetFileList( currentCategory, &filelist);
		for ( vector<string>::iterator fileindex = filelist.begin(); fileindex != filelist.end(); ++fileindex ) {
			string filepath = currentCategory + *fileindex;  // '/' + *fileindex;
			Mat image = imread( filepath );
			if ( image.empty() ) {
				continue; // maybe not an image file
			}
			vector<KeyPoint> keyPoints;
			vector<KeyPoint> keyPoints01;
			Mat descriptors;
			detector -> detect( image, keyPoints01);

			for(int i=0; i<keyPoints01.size(); i++)
			{
				KeyPoint  myPoint;

				myPoint = keyPoints01[i];

				if (myPoint.size >= MIN_KPS) keyPoints.push_back(myPoint);
			}


			extractor -> compute( image, keyPoints, descriptors );
			if ( allDescriptors.empty() ) {
				allDescriptors.create( 0, descriptors.cols, descriptors.type() );
			}
			allDescriptors.push_back( descriptors );
		}
		cout << "done processing category " << categories[index] << endl;
	}
	assert( !allDescriptors.empty() );
	cout << "build vocabulary..." << endl;
	BOWKMeansTrainer bowTrainer( wordCount );
	Mat vocabulary = bowTrainer.cluster( allDescriptors );
	cout<<"the row is "<<vocabulary.rows<<endl;
	cout<<"the col is "<<vocabulary.cols<<endl;
	cout << "done build vocabulary..." << endl;
	return vocabulary;
}

void  gzh_opencv_llc_bow_Descriptor(Mat &image, Mat &vocabulary,  vector<KeyPoint> &key_points, Mat &all_llc_descriptor)
{
#if TIME_SIFT_LLC
	double time0 = (double)cvGetTickCount();
	double uint_time = (cvGetTickFrequency() * 1000.0);
#endif

	Mat descriptors;

	Params params;
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create( params.descriptorType );
	extractor -> compute( image, key_points, descriptors );

#if TIME_SIFT_LLC
	double time1 = (double)cvGetTickCount();
#endif

	int     knn = 5;
	float  fbeta = 1e-4;

	all_llc_descriptor.create(0, VOCA_COLS, CV_32F);

#if TIME_SIFT_LLC
	double time_match_begin = (double)cvGetTickCount();
#endif

	vector<vector<DMatch> > matches;

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "FlannBased" );

	matcher -> knnMatch( descriptors, vocabulary, matches, knn );

#if TIME_SIFT_LLC
	double time_match_end = (double)cvGetTickCount();
#endif


	Mat  des_mat_r01;
	for (int icx=0; icx<descriptors.rows; icx++)
	{
		des_mat_r01 = descriptors.row(icx);

		vector<DMatch> &matchesv1 = matches[icx];

		Mat  mat_cbknn;

		mat_cbknn.release();


		for (int i=0; i<knn; i++)
		{
			Mat  mat_idx01 = vocabulary.row(matchesv1[i].trainIdx);

			mat_cbknn.push_back(mat_idx01);
		}

		Mat  ll_mat = Mat::eye(knn, knn, CV_32FC1);
		Mat  z_mat = mat_cbknn - repeat(des_mat_r01, 5, 1);
		Mat  one_mat = Mat::ones(knn, 1, CV_32FC1);
		Mat  c_mat = z_mat*z_mat.t();

		float  ftrace = trace(c_mat).val[0];

		c_mat = c_mat + ll_mat*fbeta*ftrace;

		Mat  w_mat = c_mat.inv()*one_mat;

		w_mat = w_mat/sum(w_mat).val[0];

		w_mat = w_mat.t();

		Mat llc_descriptor(1, VOCA_COLS, CV_32FC1, Scalar::all(0.0));
#if 0
		for (int i=0; i<knn; i++)
		{
			llc_descriptor.at<float>(0, matchesv1[i].trainIdx) += w_mat.at<float>(0,i);
		}
		llc_descriptor = llc_descriptor/(descriptors.rows*1.0);
#else
		//编写代码将数据放在llc_descriptor的合适位置上
		//位置的确定依赖于：matchesv1
		for (int i=0; i<knn; i++)
		{
			int code_idx = matchesv1[i].trainIdx;
			llc_descriptor.at<float>(0, code_idx) = w_mat.at<float>(0, i);
		}
#endif
		all_llc_descriptor.push_back(llc_descriptor);
	}

#if TIME_SIFT_LLC
	double time2 = (double)cvGetTickCount();
	cout<<"time using in extracting sift is "<<(time1 - time0)/uint_time<<" ms"<<endl;
	cout<<"match time of sift to vocabulary is "<<(time_match_end - time_match_begin)/uint_time<<" ms"<<endl;
	cout<<"time using of llc coding is "<<(time2 - time_match_end)/uint_time<<" ms"<<endl;
	cout<<"all time in function <gzh_opencv_llc_bow_Descriptor> is "<<(time2 - time0)/uint_time<<" ms"<<endl;
#endif
}

void max_code_response(vector<int> &points_index, Mat &all_llc_descriptor, Mat &llc_descriptor, int row_index)
{
	int points_size = points_index.size();
	int cols = all_llc_descriptor.cols;
	for(int i=0; i<cols; i++)
	{
		for(int j=0; j<points_size; j++){
			int pt_index = points_index[j];
			llc_descriptor.at<float>(row_index, i) = llc_descriptor.at<float>(row_index, i) > all_llc_descriptor.at<float>(pt_index, i) ? \
				llc_descriptor.at<float>(row_index, i) : all_llc_descriptor.at<float>(pt_index, i);
		}
	}
}

void gzh_llc_pooling(Mat &image, Mat &vocabulary,  vector<KeyPoint> &key_points, Mat &llc_descriptor, int pLevels)
{
#if TIME
	double time0 = (double)cvGetTickCount();
#endif
	//call the function to get the LLC coding result.
	Mat all_llc_descriptor;
	gzh_opencv_llc_bow_Descriptor(image, vocabulary, key_points, all_llc_descriptor);
#if TIME
	double time1 = (double)cvGetTickCount();
#endif
	//getMatPrint(all_llc_descriptor);
	//cout<<"track the size change:"<<endl;
	//cout<<"The points size is "<<key_points.size()<<endl;
	//cout<<"The descriptor size is "<<all_llc_descriptor.rows<<endl;
	//cout<<"The size of all_llc_descriptor (rows, cols): "<<all_llc_descriptor.rows<<" "<<all_llc_descriptor.cols<<endl;
	//其行数就是数据点的个数key_points.size()，列数就是码书的个数1024
	vector<int> a(pLevels, 0);
	for(int i=0; i<pLevels; i++){
		if(i==0)
			a[i] = 1;
		else
			a[i] = 2*a[i-1];
	}
	Mat pyramid(1, pLevels, CV_32S, (void*)(&(a[0])));
	Mat pBins, beta, IDXBin;
	pow(pyramid, 2, pBins);
	int tBins = (int)(sum(pBins).val[0]);

	//最终达到的尺寸是tBins * 1024
	llc_descriptor.create(tBins, VOCA_COLS, CV_32F); //CV_32F
	for(int i=0; i<tBins; i++){
		for(int j=0; j<VOCA_COLS; j++){
			llc_descriptor.at<float>(i,j) = 0.0;
		}
	}

	int bld = 0, nBins, wUnit, hUnit, xBin, yBin, idxBin;
	int pointsCount = (int)key_points.size();
	Point2f pt;
	for(int iter1=0; iter1<pLevels; iter1++)
	{
		nBins = pBins.at<int>(0, iter1);
		wUnit = (int)ceil((float)image.cols/(float)pyramid.at<int>(0, iter1));
		hUnit = (int)ceil((float)image.rows/(float)pyramid.at<int>(0, iter1));
		IDXBin.create(0, 1, CV_32FC1);
		for(int i=0; i<pointsCount; i++)
		{
			pt = key_points[i].pt;
			xBin = ceil((pt.x+esp)/wUnit);
			yBin = ceil((pt.y+esp)/hUnit);
			idxBin = (yBin - 1)*(int)pyramid.at<int>(0, iter1)+xBin-1;
			IDXBin.push_back(idxBin);
		}
		assert(IDXBin.rows==pointsCount);
		//统计各个区域的数据点的个数
		vector<vector<int> > grid_index(nBins, vector<int>(0));
		for(int i=0; i<pointsCount; i++)
		{
			grid_index[IDXBin.at<int>(i, 0)].push_back(i);
		}
		for(int iter2 = 0; iter2 < nBins; iter2++)
		{
			if(grid_index[iter2].size()==0){
				;
			}else{
				//llc_descriptor矩阵第bld行的数据
				max_code_response(grid_index[iter2], all_llc_descriptor, llc_descriptor, bld);
			}
			bld = bld + 1;
		}
	}
	//将llc_descriptor的尺寸进行变化
	llc_descriptor = llc_descriptor.reshape(0, 1);
	Mat power_llc_descriptor;
	pow(llc_descriptor, 2,power_llc_descriptor); 
	llc_descriptor = llc_descriptor/sqrt(sum(power_llc_descriptor).val[0]);
#if TIME
	double time2 = (double)cvGetTickCount();

	double uint_time = (cvGetTickFrequency() * 1000.0);
	double time_all = (time2 - time0)/uint_time;
	double sift_llc = (time1 - time0)/uint_time;
	cout<<"time using in sift and llc is "<<sift_llc<<endl;
	cout<<"time using in function gzh_llc_pooling is "<<time_all<<endl;
#endif
}

// bag of words of an image as its descriptor, not keypoint descriptors
void ComputeBowImageDescriptors( const string& databaseDir, Mat& vocabulary,
								const vector<string>& categories,
								const Ptr<FeatureDetector>& detector,
								const Ptr<DescriptorExtractor>& extractor,
								Ptr<BOWImgDescriptorExtractor>& bowExtractor,
								const string& imageDescriptorsDir,
								map<string, Mat>* samples)
{

	std::cout << "vocabulary rows cols = " << vocabulary.rows << "  " << vocabulary.cols << std::endl;

	for (int  i = 0; i != categories.size(); ++i ) 
	{
		string currentCategory = databaseDir + '/' + categories[i];
		vector<string> filelist;
		GetFileList( currentCategory+'/', &filelist);
		for ( vector<string>::iterator fileitr = filelist.begin(); fileitr != filelist.end(); ++fileitr ) {
			string descriptorFileName = imageDescriptorsDir + "/" + categories[i] + "_" + ( *fileitr ) + ".xml";

			std::cout << "bow: " << descriptorFileName << std::endl;

			FileStorage fs( descriptorFileName, FileStorage::READ );
			Mat imageDescriptor;
			if ( fs.isOpened() ) 
			{ // already cached
				fs["imageDescriptor"] >> imageDescriptor;
			}
			else 
			{
				string filepath = currentCategory + '/' + *fileitr;
				Mat image = imread( filepath );
				if ( image.empty() ) {
					continue; // maybe not an image file
				}
				vector<KeyPoint> keyPoints;
				vector<KeyPoint> keyPoints01;

				detector -> detect( image, keyPoints01 );

				for(int i=0; i<keyPoints01.size(); i++)
				{
					KeyPoint  myPoint;

					myPoint = keyPoints01[i];

					if (myPoint.size >= MIN_KPS) keyPoints.push_back(myPoint);
				}

#if 0
				opencv_llc_bow_Descriptor( image, vocabulary, keyPoints, imageDescriptor );
#else
				//替换SPM部分------------是max_pooling
				int pLevels = 3;
				gzh_llc_pooling( image, vocabulary, keyPoints, imageDescriptor, pLevels);
#endif

				fs.open( descriptorFileName, FileStorage::WRITE );
				if ( fs.isOpened() ) 
				{
					fs << "imageDescriptor" << imageDescriptor;
				}
			}
			if ( samples -> count( categories[i] ) == 0 ) 
			{
				( *samples )[categories[i]].create( 0, imageDescriptor.cols, imageDescriptor.type() );
			}
			( *samples )[categories[i]].push_back( imageDescriptor );
		}
	}
}

// bag of words of an image as its descriptor, not keypoint descriptors
void gzh_ComputeBowImageDescriptors( const string& databaseDir, Mat& vocabulary,
								const vector<string>& categories,
								/*const Ptr<FeatureDetector>& detector,*/
								const DenseFeatureDetector& detector,
								const Ptr<DescriptorExtractor>& extractor,
								Ptr<BOWImgDescriptorExtractor>& bowExtractor,
								const string& imageDescriptorsDir,
								map<string, Mat>* samples)
{

	std::cout << "vocabulary rows cols = " << vocabulary.rows << "  " << vocabulary.cols << std::endl;

	for (int  i = 0; i != categories.size(); ++i ) 
	{
		string currentCategory = databaseDir + '/' + categories[i];
		vector<string> filelist;
		GetFileList( currentCategory+'/', &filelist);
		for ( vector<string>::iterator fileitr = filelist.begin(); fileitr != filelist.end(); ++fileitr ) {
			string descriptorFileName = imageDescriptorsDir + "/" + categories[i] + "_" + ( *fileitr ) + ".xml";

			std::cout << "bow: " << descriptorFileName << std::endl;

			FileStorage fs( descriptorFileName, FileStorage::READ );
			Mat imageDescriptor;
			if ( fs.isOpened() ) 
			{ // already cached
				fs["imageDescriptor"] >> imageDescriptor;
			}
			else 
			{
				string filepath = currentCategory + '/' + *fileitr;
				Mat image = imread( filepath );
				if ( image.empty() ) {
					continue; // maybe not an image file
				}
				if(image.channels()==3)
					cvtColor(image, image, CV_BGR2GRAY);
#if USE_VLFEAT
				//get the feature of each image
				int pLevles = 2;
				imageDescriptor = getImageDescriptor(image, vocabulary, pLevles);
#else
				vector<KeyPoint> keyPoints;
				vector<KeyPoint> keyPoints01;

				detector.detect( image, keyPoints01 );

				for(int i=0; i<keyPoints01.size(); i++)
				{
					KeyPoint  myPoint;

					myPoint = keyPoints01[i];

					if (myPoint.size >= MIN_KPS) keyPoints.push_back(myPoint);
				}


				//imageDescriptor = Mat::zeros(1, VOCA_COLS, CV_32F);
#if 0
				opencv_llc_bow_Descriptor( image, vocabulary, keyPoints, imageDescriptor );
#else
				//替换SPM部分------------是max_pooling
				int pLevels = 3;
				gzh_llc_pooling( image, vocabulary, keyPoints, imageDescriptor, pLevels);
#endif
#endif

				//std::cout << "imageDescriptor rows cols = " << imageDescriptor.rows << "  "
				//<< imageDescriptor.cols << std::endl;

				fs.open( descriptorFileName, FileStorage::WRITE );
				if ( fs.isOpened() ) 
				{
					fs << "imageDescriptor" << imageDescriptor;
				}
			}
			if ( samples -> count( categories[i] ) == 0 ) 
			{
				( *samples )[categories[i]].create( 0, imageDescriptor.cols, imageDescriptor.type() );
			}
			( *samples )[categories[i]].push_back( imageDescriptor );
		}
	}
}


void LoadSvms(const string& svms_dir, map<string, CvSVM*>&  svms_map)
{
	vector<string>  svms_fns;
	GetFileList( svms_dir + '/', &svms_fns );
	for ( vector<string>::iterator itr = svms_fns.begin(); itr != svms_fns.end();  itr++)
	{
		string   svm_fn = *itr;
		int  n = svm_fn.find(".xml");
		string   name_ic;
		name_ic.assign(svm_fn, 0, n);
		CvSVM  *psvm = new CvSVM;
		string svmFileName = svms_dir + "/" + svm_fn;

		FileStorage fs( svmFileName, FileStorage::READ );
		if ( fs.isOpened() )
		{
			fs.release();
			psvm->load( svmFileName.c_str() );
			svms_map.insert(pair<string, CvSVM*>(name_ic, psvm));
		}
		else
		{
			std::cout << "svm : " << svmFileName << " can not load " << std::endl;
			exit(-1);
		}
		std::cout << name_ic << " svm :  " << svmFileName << std::endl;
	}
}

void classifier_hand( Mat& image, map<string, CvSVM*>& svms_map,int& category )
{
	Params params;
	
#if !DENSE_SIFT
	Ptr<FeatureDetector> detector = FeatureDetector::create( params.detectorType );
#else
	DenseFeatureDetector detector(FEATURE_SCALE);
#endif
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create( params.descriptorType );

	Mat vocabulary;
	string vocabularyFile = "..\\gaor\\vocabulary_gaor.xml.gz";//resultDir + '/' + kVocabularyFile;
	FileStorage fs( vocabularyFile, FileStorage::READ );
	if ( fs.isOpened() ) {
		fs["vocabulary"] >> vocabulary;
	} 
	else 
	{
		return ;
	}

	int sample_siz = 60;
	int stepX = 6;
	int stepY = 6;
	int sift_dim = 128;
	filter = initialize_vldsift_filter(sample_siz, sample_siz, stepX, stepY);
	//num 是图像中SIFT特征点个数
	int num_g = vl_dsift_get_keypoint_num(filter);
	//const VlDsiftKeypoint *
	keyPoints = new VlDsiftKeypoint[num_g];
	//Mat des;
	des.create(num_g, sift_dim, CV_32FC1);

	int pLevels = 2;
	int knn = 5;
	Mat queryDescriptor;
	gzh_vlfeat_sift_llc_max_pooling(image, vocabulary, filter, keyPoints, des, queryDescriptor, knn, pLevels);


	int sign = 0; //sign of the positive class
	float confidence = -FLT_MAX;

	vector<int> lable(ClASS, 0);

	int i = 0;
	for (map<string, CvSVM*>::const_iterator itp = svms_map.begin(); itp != svms_map.end(); ++itp )
	{

		CvSVM  *psvm = itp->second;
		float response =psvm->predict(queryDescriptor);
		lable[i] = (int)response;
		i++;
	}
	vector<int> num(ClASS, 0); 
	int maxnum=0;

	for(int j=0; j<ClASS; j++)
	{

		num[lable[j]]++;
	}

	maxnum = 0;
	category = 0;
	for (int k=0; k<ClASS; k++)
	{
		if(num[k]>maxnum)
		{
			maxnum = num[k];
			category = k;
		}
	}
}

Mat gzh_BuildVocabulary( const string& databaseDir,
					const vector<string>& categories,
					const DenseFeatureDetector& detector,
					/*const Ptr<FeatureDetector>& detector,*/
					const Ptr<DescriptorExtractor>& extractor,
					int wordCount)
{
	Mat allDescriptors;
	for ( int index = 0; index != categories.size(); ++index ) {
		cout << "processing category " << categories[index] << endl;
		string currentCategory = databaseDir +'/' + categories[index] + '/';
		vector<string> filelist;
		GetFileList( currentCategory, &filelist);
		for ( vector<string>::iterator fileindex = filelist.begin(); fileindex != filelist.end(); ++fileindex ) {
			string filepath = currentCategory + *fileindex;  // '/' + *fileindex;
			Mat image = imread( filepath );
			if ( image.empty() ) {
				continue; // maybe not an image file
			}
			if(image.channels()==3)
				cvtColor(image, image, CV_BGR2GRAY);
#if USE_VLFEAT
			Mat descriptors = gzh_vlfeat_sift(image);
#else
			vector<KeyPoint> keyPoints;
			vector<KeyPoint> keyPoints01;
			Mat descriptors;
			detector.detect( image, keyPoints01);

			for(int i=0; i<keyPoints01.size(); i++)
			{
				KeyPoint  myPoint;

				myPoint = keyPoints01[i];

				if (myPoint.size >= MIN_KPS) keyPoints.push_back(myPoint);
			}
			extractor -> compute( image, keyPoints, descriptors );
#endif
			if ( allDescriptors.empty() ) {
				allDescriptors.create( 0, descriptors.cols, descriptors.type() );
			}
			allDescriptors.push_back( descriptors );
		}
		cout << "done processing category " << categories[index] << endl;
	}
	assert( !allDescriptors.empty() );
	cout << "build vocabulary..." << endl;
	BOWKMeansTrainer bowTrainer( wordCount );
	Mat vocabulary = bowTrainer.cluster( allDescriptors );
	cout<<"the row is "<<vocabulary.rows<<endl;
	cout<<"the col is "<<vocabulary.cols<<endl;
	cout << "done build vocabulary..." << endl;
	return vocabulary;
}

void help( const char* progName )
{
	std::cout << "OpenCV LLC BOW ..." << std::endl << std::endl;

	std::cout << "train: " << progName << " [train] [databaseDir] [resultDir] " << std::endl;
	std::cout << "  example: " << progName << " train  ../data/train/  ../data/result/ " << std::endl;

	std::cout << std::endl;

	std::cout << "test: " << progName << " [test] [sample_name] [test_dir] [svms_dir] [vocabulary_file] " << std::endl;
	std::cout << "  example: " << progName << " test  sunflower  ../data/imgs/sunflower  ../data/result/svms  ../data/result/vocabulary.xml.gz" << std::endl;
}

void MakeDir( const string& filepath )
{

	char path[MAX_PATH];

	strncpy(path, filepath.c_str(),  MAX_PATH);

#ifdef _WIN32
	mkdir(path);
#else
	mkdir(path, 0755);
#endif
}

void ListDir( const string& directory, vector<string>* entries)
{
	HANDLE hlistfile;
	WIN32_FIND_DATA find_file_data;
	CString dirpath(directory.c_str());
	CString dirpathall=dirpath+"*";
	hlistfile=FindFirstFile(dirpathall,&find_file_data);
	if(hlistfile==INVALID_HANDLE_VALUE)
		exit(-1);
	do{
		if(strcmp((char*)find_file_data.cFileName,".") != 0 && strcmp((char*)find_file_data.cFileName,"..") != 0)
		{
			CString filename=dirpath ;
			filename=find_file_data.cFileName;
			char file_name[200] = "";
			sprintf(file_name, "%s", (LPSTR)(LPCTSTR)(filename));
			printf("%s\n", file_name);
			string file_copy(file_name);
			entries->push_back(file_copy);
		}
		if(!FindNextFile(hlistfile,&find_file_data))
			break;
	}while(1);
}

void GetFileList( const string& directory, vector<string>* filelist )
{
	ListDir( directory, filelist);
}