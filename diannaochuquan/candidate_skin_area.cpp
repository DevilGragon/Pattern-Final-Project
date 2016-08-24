#include "stdafx.h"

#include "highgui.h"
#include "cv.h"
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

void SkinCrCbDetect(IplImage* src,IplImage* dst)
{
	//This program is used to get the binary image of the src and save 
	//in the image called dst.
	cvSmooth(src,src);
	IplImage* YCrCb=cvCreateImage(cvGetSize(src),src->depth,src->nChannels);
	cvCvtColor(src,YCrCb,CV_BGR2YCrCb);
	//cvCvtColor(src,YCrCb,CV_BGR2YUV);
	//Then I will threshold the image with the empirical value of the Cr and Cb.
	int i,j;
	int height=src->height;
	int width=src->width;
	uchar* ptr_src,*ptr_dst;
	for(i=0;i<height;i++)
	{
		ptr_src=(uchar*)(YCrCb->imageData+i*YCrCb->widthStep);
		ptr_dst=(uchar*)(dst->imageData+i*dst->widthStep);
		for(j=0;j<width;j++)
		{
			//Thresholding.
			if((133<=ptr_src[3*j+1]&&ptr_src[3*j+1]<=173)&&(77<=ptr_src[3*j+2]&&ptr_src[3*j+2]<=127))
			//if((95<=ptr_src[3*j+1]&&ptr_src[3*j+1]<=128)&&(134<=ptr_src[3*j+2]&&ptr_src[3*j+2]<=182))
				ptr_dst[j]=255;
			else
				ptr_dst[j]=0;


		}
	}
	//Then I will use the morphologic operation.
	cvErode(dst,dst,NULL,2);
	cvDilate(dst,dst,NULL,1);
	//cvShowImage("gray",dst);
	cvReleaseImage(&YCrCb);
}

//a's area is bigger than b
bool JudgeContainRectangle(Rect& a, Rect& b )
{
	if(a.x>b.x)
		return 0;//b not in a
	if(a.y>b.y)
		return 0;
	if((a.x+a.width)<(b.x+b.width))
		return 0;
	if((a.y+a.height)<(b.y+b.height))
		return 0;
	return 1;//b in a
}


void  CandidateSkinArea(IplImage* src,CvMemStorage* store,vector<Rect>& candidate_SkinArea)
{

	CvSeq * contour = 0;
	cvFindContours( src, store, &contour, sizeof(CvContour),\
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	Rect SkinArea;
	Rect TmpSkinArea;
	int i,j,flag;
	int x,y;
	candidate_SkinArea.clear();
	for(;contour;contour = contour->h_next)
	{
		//Control the size of the rectangle
		SkinArea=cvBoundingRect( contour, 0 );
		if ((SkinArea.height<30 )|| (SkinArea.width<30))
			continue;
		x=SkinArea.x+SkinArea.width/2;
		y=SkinArea.y+SkinArea.height/2;

		SkinArea.width=min((SkinArea.width+20),src->width);
		SkinArea.height=min((SkinArea.height+20),src->height);
		if(SkinArea.width>SkinArea.height)
			SkinArea.width=SkinArea.height;

		SkinArea.x=max((x-SkinArea.width/2),0);
		SkinArea.y=max((y-SkinArea.height/2),0);
		if((SkinArea.x+SkinArea.width)>src->width)
			SkinArea.width = src->width-SkinArea.x;
		if((SkinArea.y+SkinArea.height)>src->height)
			SkinArea.height = src->height-SkinArea.y;
		if(candidate_SkinArea.size()==0)
		{
			candidate_SkinArea.push_back(SkinArea);
			continue;
		}else
		{
			flag=0;
			for(i=0;i<candidate_SkinArea.size();i++)
			{
				TmpSkinArea=candidate_SkinArea[i];
				if ((SkinArea.height*SkinArea.width)<TmpSkinArea.area())
				{
					if(JudgeContainRectangle(TmpSkinArea,SkinArea)==1)
					{
						flag=1;					
						break;
					}
					else
						continue;
				}
				else 
					break;					
			}
			if(flag)
				continue;
			candidate_SkinArea.insert(candidate_SkinArea.begin()+i,SkinArea);
			for(j=i+1;j<candidate_SkinArea.size();j++)
			{
				TmpSkinArea=candidate_SkinArea[j];
				if(JudgeContainRectangle(SkinArea,TmpSkinArea)==1)
					candidate_SkinArea.erase(candidate_SkinArea.begin()+j);	
			}
		}		
	}
}

void ChangeArea( CvRect& candidateRect1, IplImage* image)
{
	candidateRect1.x=max((candidateRect1.x-5),0);
	candidateRect1.y=max((candidateRect1.y-5),0);
	candidateRect1.width=candidateRect1.width+10;
	candidateRect1.height=candidateRect1.height+10;

	if((candidateRect1.x+candidateRect1.width)>image->width)
		candidateRect1.width = (int)(image->width-candidateRect1.x);
	if((candidateRect1.y+candidateRect1.height)>image->height)
		candidateRect1.height = (int)(image->height-candidateRect1.y);

}

