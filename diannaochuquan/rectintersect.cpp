#include "stdafx.h"

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "stdio.h"


bool rectarea(CvRect rect0,CvRect rect1,float thresh)
{
	bool flag=false;

	int x0,y0,w0,h0;
	int x,y,w,h;
	int dx=0,dy=0,cx=0,cy=0;     //dx为两矩形x坐标差，cx,cy交叉区域的中心坐标
	int cw=0,ch=0;               //交叉区域的宽高
	float arearate=0;            //交叉区域面积与原矩形面积之比
	int yh=0,xw=0,yh0=0,xw0=0;   //中间变量
	//puts("please input the attribute of the rectangle.\n ");
	//scanf("%d %d %d %d",&x,&y,&w,&h);
	//printf("x= %d y= %d w= %d h= %d \n",x,y,w,h);
	x0=rect0.x;
	y0=rect0.y;
	w0=rect0.width;
	h0=rect0.height;

	x=rect1.x;
	y=rect1.y;
	w=rect1.width;
	h=rect1.height;

	yh=y+h;
	xw=x+w;
	yh0=y0+h0;
	xw0=x0+w0;

	if(!((y>(y0-h)&&y<yh0)&&(x>(x0-w)&&x<xw0)))
	{
		//puts("no intersection!\n");
		return false;
	}

	//交叉区域x坐标
	if(x<x0)
	{
		cx=x0;
		cw=(xw>=xw0)?w0:(xw-x0);
	}
	else
	{
		cx=y0;
		cw=(xw>=xw0)?(xw0-x):w;
	}
	//交叉区域y坐标
	if(y<y0)
	{
		cy=y0;
		ch=(yh>=yh0)?h0:(yh-y0);
	}
	else
	{
		cy=y;
		ch=(yh>=yh0)?(yh0-y):h;
	}

	//交叉区域面积
	arearate=(float)(cw*ch*1.0)/(w0*h0);
	//显示交叉区域参数
	//printf("cx= %d cy= %d cw= %d ch= %d \n",cx,cy,cw,ch);
	//printf("arearate = %f%% \n",arearate*100);

	if(arearate<thresh)
		return false;
	else
		return true;
}

