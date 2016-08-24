#include "stdafx.h"

#include "vl/dsift.h"
#include "vl/sift.h"
#include "highgui.h"
#include "cv.h"
#include <string>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

#define MATRIX_WRITE 0
#define VLFEAT_NORM 1
#define esp 1e-10

#define TIC  (double)cvGetTickCount()
#define TICKS_PER_MS (cvGetTickFrequency()*1000.0)
#define TOC(num)  (TIC - (double)num)/TICKS_PER_MS

#define TIME_DIRECT 0
#define RESIZE 0

void getTime(string file, double time)
{
	fstream f;
	f.open(file.c_str(), ios::app);
	if(f.is_open()){
		f<<time<<" ";
		f.close();
	}
}

void getMatToFile(string file, Mat data)
{
	fstream f;
	f.open(file.c_str(), ios::out);
	if(f.is_open()){
		f.clear();
		f<<data;
		f.close();
	}
}

VlDsiftFilter * initialize_vldsift_filter(int width, int height, int stepX, int stepY)
{
	//build the vldsiftfilter 
	VlDsiftFilter *dSIFT_filter = vl_dsift_new(width, height);
	vl_dsift_set_steps(dSIFT_filter, stepX, stepY);
	return dSIFT_filter;
}

void vlfeatSIFT_points_extract_same_filter(Mat &src, VlDsiftFilter *dSIFT_filter)
{
	//1. image processing 
	if(src.channels() == 3)
		cvtColor(src, src, CV_BGR2GRAY);
#if RESIZE
	int image_size = 60;
	resize(src, src, Size(image_size, image_size));
#endif
	int width = src.cols, height = src.rows;
	//2. extract the points and descriptors
	//但是前提是对于60*60 的图像块，必须首先建立好相应的VlDsiftFilter
	vl_sift_pix *ImageData = new vl_sift_pix[height*width];
	for (int i=0; i<height; i++)     //y
	{
		for (int j=0; j<width; j++)  //x
		{
			ImageData[i*width + j]= (vl_sift_pix)(src.at<uchar>(i,j));
		}
	}
#if TIME_DIRECT
	double time0 = TIC;
#endif
	vl_dsift_process(dSIFT_filter, ImageData);
#if TIME_DIRECT
	double time_consume = TOC(time0);
	getTime("vlfeat.txt", time_consume);
	cout<<"time using "<<time_consume<<" ms"<<endl;
#endif
	delete[] ImageData;
}

bool idx_exist(int idx, int *index, int length)
{
	if(length!=0){
		for(int i=0; i<length; i++){
			if(index[i] == idx){
				return true;
			}
		}
	}
	return false;
}

void find_indexs_of_minimum(float *data, int data_dim, int *index, int index_dim)
{
	//This function is used to find the <index_dim>s num of <float* data> which points to
	//data_dim data, and put the indexs of them into the address call <int *index>.
	int minimum_idx;
	float minimum;
	for(int found_index = 0; found_index<index_dim; found_index++){
		//It means that we already find <found_index> values.
		minimum = FLT_MAX;
		for(int i=0; i<data_dim; i++){
			if(idx_exist(i, index, found_index))
				continue;
			else{
				if(data[i] < minimum){
					minimum = data[i];
					minimum_idx = i;
				}
			}//end of if-else
		}//end of for data loop
		index[found_index] = minimum_idx;
	}
}


extern void max_code_response(vector<int> &points_index, Mat &all_llc_descriptor, Mat &llc_descriptor, int row_index);


void gzh_vlfeat_sift_llc_max_pooling(Mat &src, Mat &vol, VlDsiftFilter *filter, 
									 const VlDsiftKeypoint *keyPoints, Mat &des,
									 Mat &descriptor_after_pooling, int knn, int pLevels)
{
	//This function is used to extract the features of the image.
	//We need to allocate some space for the variables called des and 
	//keyPoints which is decided by the <filter>, the final descriptor of 
	//each image is called <descriptor_after_pooling> which is achieved by
	//vlfeat sift extraction + LLC + max-pooling.
	//extract the descrptor and points
	//Step 1: get the descriptors of this image.
	int num = vl_dsift_get_keypoint_num(filter);
	int vol_num = vol.rows;
	vlfeatSIFT_points_extract_same_filter(src, filter);
	keyPoints = vl_dsift_get_keypoints(filter);

#if 0
	Mat gzh_pts(num, 2, CV_32F, Scalar::all(0.0));
	for(int i=0; i<num; i++){
		gzh_pts.at<float>(i, 0) = keyPoints[i].x;
		gzh_pts.at<float>(i, 1) = keyPoints[i].y;
	}
	getMatToFile("LLC/new_keypoints.txt", gzh_pts);
#endif
	//cout<<"points num is "<<num<<endl;
	des.data = (uchar*)(vl_dsift_get_descriptors(filter));

	//Step 2: LLC coding of the descriptors of this image to vocabulary.
	//LLC coding to get the image descriptor
	//int knn = 5;
	float  fbeta = 1e-4;
#if VLFEAT_NORM
	Mat big_one_mat(num, vol_num, CV_32FC1, Scalar::all(2.0));
	Mat D = big_one_mat - 2 * des * vol.t();
#else
	Mat desX;
	Mat volB;
	pow(des, 2, desX);
	pow(vol, 2, volB);
#if MATRIX_WRITE
	getMatToFile("LLC/des.txt", des);
	getMatToFile("LLC/vol.txt", vol);
	getMatToFile("LLC/desX.txt", desX);
	getMatToFile("LLC/volB.txt", volB);
#endif

	//num is the number of the points in this image
	//int vol_num = vol.rows;
	reduce(desX, desX, 1, CV_REDUCE_SUM);
	reduce(volB, volB, 1, CV_REDUCE_SUM);
#if MATRIX_WRITE
	getMatToFile("LLC/desXX.txt", desX);
	getMatToFile("LLC/volBB.txt", volB);
#endif
	Mat desX_extend, volB_extend;
	repeat(desX, 1, vol_num, desX_extend);
	repeat(volB.t(), num, 1, volB_extend);
	Mat D = desX_extend + volB_extend - 2 * des * vol.t();
#endif
#if MATRIX_WRITE
	getMatToFile("LLC/new_D.txt", D);
#endif
	Mat IDX(num, knn, CV_32SC1, Scalar::all(0));
	float *row_ptr_D;
	int *row_ptr_IDX;
	for(int i=0; i<num; i++){
		//find the index of the maximum <knn> values.
		//row pointer, the num in a row is row of codebook(vol_num).
		row_ptr_D = D.ptr<float>(i);
		row_ptr_IDX=IDX.ptr<int>(i);
		find_indexs_of_minimum(row_ptr_D, vol_num, row_ptr_IDX, knn);
	}
#if MATRIX_WRITE
	getMatToFile("LLC/IDX.txt", IDX);
#endif
	//Then we find the <knn>s nearest neighbours in the coodbook of descriptors.
	Mat ll_mat = Mat::eye(knn, knn, CV_32FC1);
	Mat all_llc_descriptor;
	all_llc_descriptor.create(0, vol_num, CV_32FC1);

	Mat  des_mat_r01;
	for (int icx=0; icx<num; icx++)
	{
		des_mat_r01 = des.row(icx);
		Mat  mat_cbknn;
		mat_cbknn.release();
		for (int i=0; i<knn; i++)
		{
			Mat  mat_idx01 = vol.row(IDX.at<int>(icx, i));
			mat_cbknn.push_back(mat_idx01);
		}
		//Mat  ll_mat = Mat::eye(knn, knn, CV_32FC1);
		Mat  z_mat = mat_cbknn - repeat(des_mat_r01, 5, 1);
		Mat  one_mat = Mat::ones(knn, 1, CV_32FC1);
		Mat  c_mat = z_mat*z_mat.t();
		float  ftrace = trace(c_mat).val[0];
		c_mat = c_mat + ll_mat*fbeta*ftrace;
		Mat  w_mat = c_mat.inv()*one_mat;
		w_mat = w_mat/sum(w_mat).val[0];
		w_mat = w_mat.t();
		Mat llc_descriptor(1, vol_num, CV_32FC1, Scalar::all(0.0));
		//编写代码将数据放在llc_descriptor的合适位置上
		//位置的确定依赖于：matchesv1
		for (int i=0; i<knn; i++)
		{
			int code_idx = IDX.at<int>(icx, i);
			llc_descriptor.at<float>(0, code_idx) = w_mat.at<float>(0, i);
		}
#if MATRIX_WRITE
		getMatToFile("LLC/new_w_mat.txt", w_mat);
#endif
		all_llc_descriptor.push_back(llc_descriptor);
	}
	//Step 3: after the LLC coding method,  we need to use max-pooling method 
	//to get the final descriptor.
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

	//最终达到的尺寸是tBins * codebook_entry
	descriptor_after_pooling.create(tBins, vol_num, CV_32F);
	for(int i=0; i<tBins; i++){
		for(int j=0; j<vol_num; j++){
			descriptor_after_pooling.at<float>(i,j) = 0.0;
		}
	}


	//beta = Mat::zeros(vocabulary.cols, tBins, CV_32FC1);
	int bld = 0, nBins, wUnit, hUnit, xBin, yBin, idxBin;
	int pointsCount = num;//(int)key_points.size();
	Point2f pt;
	for(int iter1=0; iter1<pLevels; iter1++)
	{
		nBins = pBins.at<int>(0, iter1);
		wUnit = (int)ceil((float)src.cols/(float)pyramid.at<int>(0, iter1));
		hUnit = (int)ceil((float)src.rows/(float)pyramid.at<int>(0, iter1));
		IDXBin.create(0, 1, CV_32FC1);
		for(int i=0; i<pointsCount; i++)
		{
			//pt = key_points[i].pt;
			pt.x = keyPoints[i].x;
			pt.y = keyPoints[i].y;
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
			//bld = bld + 1;
			if(grid_index[iter2].size()==0){
				;//continue;
			}else{
				//llc_descriptor矩阵第bld行的数据
				//void max_code_response(vector<int> &points_index, Mat &all_llc_descriptor, Mat &llc_descriptor, int row_index)
				max_code_response(grid_index[iter2], all_llc_descriptor, descriptor_after_pooling, bld);
			}
			bld = bld + 1;
		}
	}
	//将llc_descriptor的尺寸进行变化
	descriptor_after_pooling = descriptor_after_pooling.reshape(0, 1);
	Mat power_llc_descriptor;
	pow(descriptor_after_pooling, 2,power_llc_descriptor); 
	descriptor_after_pooling = descriptor_after_pooling/sqrt(sum(power_llc_descriptor).val[0]);
}

Mat gzh_vlfeat_sift(Mat &src)
{
	//This function is used to extract the features of the image.
	//Step 1: build the filter
	int width = src.cols;
	int height = src.rows;
	int stepX = 6;
	int stepY = 6;
	int sift_dim = 128;
	VlDsiftFilter *filter = initialize_vldsift_filter(width, height, stepX, stepY);
	int num = vl_dsift_get_keypoint_num(filter);
	vlfeatSIFT_points_extract_same_filter(src, filter);
	//Step 2: get the descriptor of this image.
	Mat des;
	des.create(num, sift_dim, CV_32FC1); 
	const float *ptr_des = (vl_dsift_get_descriptors(filter));
	for(int i=0; i<num; i++){
		for(int j=0; j<sift_dim; j++){
			des.at<float>(i, j) = ptr_des[i * sift_dim + j];
		}//end of inner for loop
	}//end of outer for loop
	return des;
}

Mat getImageDescriptor(Mat &src, Mat &vol, int pLevles)//, Mat &descriptor_after_pooling)
{
	//This function can be used in the train of the system to get the description of the image.
	int width = src.cols;
	int height = src.rows;
	int stepX = 6;
	int stepY = 6;
	int sift_dim = 128;
	VlDsiftFilter *filter = initialize_vldsift_filter(width, height, stepX, stepY);
	//allocate the space
	//num is the number of the points in this image
	int num = vl_dsift_get_keypoint_num(filter);
	const VlDsiftKeypoint *keyPoints = new VlDsiftKeypoint[num];
	Mat des;
	des.create(num, sift_dim, CV_32FC1); 
	int knn = 5;
	//int pLevles = 3;

	Mat descriptor_after_pooling;
	gzh_vlfeat_sift_llc_max_pooling(src, vol, filter, keyPoints, 
		des, descriptor_after_pooling, knn, pLevles);
	delete [] keyPoints;
	vl_dsift_delete(filter);
	return descriptor_after_pooling;
}