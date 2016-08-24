//#include "stdafx.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#define MAXRECT 1000

#ifndef _RECTLOAREA_H
	#define _RECTLOAREA_H
	struct rectloc
	{
		bool flage;
		int x;
		int y;
		int width;
		int height;
		rectloc *next;
	};

	struct initstruct
	{
		int handnum;
		int totalrect;
		float thresh;
		int qjl[MAXRECT];
		int zj_num[MAXRECT];
		float zjl[MAXRECT];
		float qj;
		float zj;
		rectloc *head;
		rectloc *pt;
		FILE *tempfilez;
		FILE *tempfileq;
		double time_sum;
	};
#endif
	

void RectLocatlist(rectloc **head,rectloc *p0,rectloc **last);
void releaselist(rectloc *head);
void listprint(rectloc *head);

bool rectarea(CvRect rect0,CvRect rect1,float thresh);
bool readrectloc(char *filename,rectloc **head);

bool initialdata(struct initstruct *D,char *filename);
int	getintersect(struct initstruct *D,rectloc **pt,int result,int h,CvSeq *faces, double time);
void endaddest(struct initstruct *D,int h);