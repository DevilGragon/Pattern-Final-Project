#include "stdafx.h"

#include <tchar.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

using namespace cv;
using namespace std;

extern "C"{
#include <vl/generic.h>
#include <vl/stringop.h>
#include <vl/pgm.h>
#include <vl/sift.h>
#include <vl/getopt_long.h>
};

int main_vlfeat_test(int argc, _TCHAR* argv[])
{
    VL_PRINT ("Hello world!\n") ;
    char *ImagePath="lena.jpg";
    IplImage *Image=cvLoadImage(ImagePath,0);
    //  int min=0;
    //  min=Image->width>Image->height?Image->height:Image->width;
    int noctaves=4,nlevels=2,o_min=0;
    // noctaves=(int)(log(min)/log(2));
    vl_sift_pix *ImageData=new vl_sift_pix[Image->height*Image->width];
    unsigned char *Pixel;
    for (int i=0;i<Image->height;i++)
    {
        for (int j=0;j<Image->width;j++)
        {
            Pixel=(unsigned char*)(Image->imageData+i*Image->width+j);
            ImageData[i*Image->width+j]=*(Pixel);
        }
    }
    VlSiftFilt *SiftFilt=NULL;
    SiftFilt=vl_sift_new(Image->width,Image->height,noctaves,nlevels,o_min);
    int KeyPoint=0;
    int idx=0;
    if (vl_sift_process_first_octave(SiftFilt,ImageData)!=VL_ERR_EOF)
    {
        while (1)
        {
            //计算每组中的关键点
            vl_sift_detect(SiftFilt);
            //遍历并绘制每个点
            KeyPoint+=SiftFilt->nkeys;
            VlSiftKeypoint *pKeyPoint=SiftFilt->keys;
            for (int i=0;i<SiftFilt->nkeys;i++)
            {
                VlSiftKeypoint TemptKeyPoint=*pKeyPoint;
                pKeyPoint++;
                cvDrawCircle(Image,cvPoint(TemptKeyPoint.x,TemptKeyPoint.y),TemptKeyPoint.sigma/2,CV_RGB(255,0,0));
                idx++;
                //计算并遍历每个点的方向
                double angles[4];
                int angleCount=vl_sift_calc_keypoint_orientations(SiftFilt,angles,&TemptKeyPoint);
                for (int j=0;j<angleCount;j++)
                {
                    double TemptAngle=angles[j];
                    printf("%d: %f\n",j,TemptAngle);
                    //计算每个方向的描述
                    float *Descriptors=new float[128];
                    vl_sift_calc_keypoint_descriptor(SiftFilt,Descriptors,&TemptKeyPoint,TemptAngle);
                    int k=0;
                    while (k<128)
                    {
                        printf("%d: %f",k,Descriptors[k]);
                        printf("; ");
                        k++;
                    }

                    printf("\n");
                    delete []Descriptors;
                    Descriptors=NULL;
                }
            }
            //下一阶
            if (vl_sift_process_next_octave(SiftFilt)==VL_ERR_EOF)
            {
                break;
            }
            //free(pKeyPoint);
            KeyPoint=NULL;
        }
    }
    vl_sift_delete(SiftFilt);
    delete []ImageData;
    ImageData=NULL;
    cvNamedWindow("Source Image",1);
    cvShowImage("Source Image",Image);
    cvWaitKey(0);
    cvReleaseImage(&Image);
    cvDestroyAllWindows();
    return 0;
}