#include "stdafx.h"

#include "precomp.hpp"
#include <stdio.h>
#include <iostream>
#include <omp.h>
using namespace std;
//#define T_PI 3.1415926535897932384626433832795  
//#define SMALL_NUM 0.000001 

#define T_PI 3.142  
#define T_T_PI 3.141
#define SMALL_NUM 0.000001          
short bin_table[511][511] = {0};
float mag_table[511][511] = {0.0F};
short mapbuf[3000] = {0};//w+h+4 if img is bigger, we need to change it
short *xmap = NULL;
short *ymap = NULL;


void initialize_lookup_table()
{
	int x, y;
	float angle;
	float anglebin;
	float angleScale = 9/T_PI;
	float magnitude;
	int index;
	int binNum = 9;

	for(x=-255;x<=255;x++)
		for(y=-255;y<=255;y++)
			{	 
				magnitude = (float)sqrt((x*x+y*y)*1.0);
				angle = atan2((float)y,(float)x);
				
				if (angle < 0)
				{
					angle += T_PI;
					if (angle > T_T_PI)
						angle = T_T_PI;
				}

				anglebin = angle*angleScale;

				assert(anglebin>=0&&anglebin<9);
				
				index = (int)floor(anglebin);
				
				assert(index < binNum && index >= 0); 
				
				bin_table[x+255][y+255] = index; 
				
				mag_table[x+255][y+255] = magnitude;  
				
			}
			
//	printf("Hog table initialized!\n");

}


void initialize_mapbuf(int w, int h)
{
	 
	for(int i = 1; i <= w; i++)
		mapbuf[i] = i - 1;

	for(int i = w + 3, j = 0; i <= w + 2 +h; i++,j++)
		mapbuf[i] = j;

	mapbuf[0] = 0;
	mapbuf[w+1] = w - 1;
	mapbuf[w+2] = 0;
	mapbuf[w+3+h] = h - 1;

	xmap = mapbuf + 1;
	ymap = mapbuf + w + 3;

//	printf("Mapbuf initialized!\n");
}


void tHOGIntegral(const CvMat* src, CvMat* grad, int binNum)
{
	
	CvMat t_grad = cvMat( src->height, src->width, CV_32FC1, 
					cvAlloc( sizeof( float ) * src->height * src->width ) );

	CvMat t_bin = cvMat( src->height, src->width, CV_16SC1, 
					cvAlloc( sizeof( short ) * src->height * src->width ) );

	uchar* data_img = src->data.ptr;
	int srcstep=src->step;
	int srcw=src->width;
	int srch=src->height;

	float* t_grad_data = (float*)t_grad.data.ptr;
	short* t_bin_data = (short*)t_bin.data.ptr;
	
	int x,y;
	int index;
	int dx,dy;

	for( y = 0; y < srch; y++ )
    {
        const uchar* currPtr = data_img + srcstep*ymap[y];
        const uchar* prevPtr = data_img + srcstep*ymap[y-1];
        const uchar* nextPtr = data_img + srcstep*ymap[y+1];
 
        for( x = 0; x < srcw; x++ )
        {
            dx = currPtr[xmap[x+1]] - currPtr[xmap[x-1]];
            dy = nextPtr[xmap[x]] - prevPtr[xmap[x]];

			*(t_bin_data++) = bin_table[dx+255][dy+255];		
			*(t_grad_data++) = mag_table[dx+255][dy+255];
        }
	}

	float* hog_data;
	t_grad_data = (float*)t_grad.data.ptr;
	t_bin_data = (short*)t_bin.data.ptr;

	int hogStep = (int)( grad[0].step / sizeof(float) );
	int gradStep = (int)( t_grad.step / sizeof(float) );
	int binStep = (int)( t_bin.step / sizeof(short) );


    for( int binIdx = 0; binIdx < binNum; binIdx++ )//对幅值进行分解，每个hog_grad[i]各自求积分图
    {
		hog_data = (float*)grad[binIdx].data.ptr;

		t_grad_data = (float*)t_grad.data.ptr;

		t_bin_data = (short*)t_bin.data.ptr;

		memset( hog_data, 0, grad[0].width * sizeof(hog_data[0]) );
        
		hog_data += hogStep + 1;

		for( y = 0; y < t_bin.height; y++ )
        {
            hog_data[-1] = 0.f;	
            float strSum = 0.f;

            for( x = 0; x < t_bin.width; x++ )
            {
                if( t_bin_data[x] == binIdx )
                    strSum += t_grad_data[x];
                hog_data[x] = hog_data[-hogStep + x] + strSum;
            }
            hog_data += hogStep;
            t_bin_data += binStep;
            t_grad_data += gradStep;
        }
    }

	cvFree( &(t_grad.data.ptr) );
	cvFree( &(t_bin.data.ptr) );
}