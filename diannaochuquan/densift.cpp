#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;



int main_densift(int argc, char** argv)
{
   Mat img = imread("1.bmp"), img1;;
   
   cvtColor( img, img1, CV_BGR2GRAY );

   DenseFeatureDetector dense;
   int a=1;
   DenseFeatureDetector test(16.0f);

   vector<KeyPoint> key_points;
   Mat output_img;

   dense.detect(img1,key_points,Mat());
   drawKeypoints(img, key_points, output_img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
   cout<<"the size of points is "<<key_points.size()<<endl;
   cout<<"the size of the image is "<<img.rows*img.cols<<endl;

   namedWindow("DENSE");
   imshow("DENSE", output_img);
   waitKey(0);

   return 0;
}