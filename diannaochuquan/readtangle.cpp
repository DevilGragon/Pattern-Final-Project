// readtangle.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "rectloarea.h"



//#define MAX 21
#define LEN sizeof(rectloc)

//链表生成
void RectLocatlist(rectloc **head,rectloc *p0,rectloc **last)
{
	if((*head)==NULL)
	{
		(*head)=p0;
		(*last)=p0;
	}
	else
	{
		(*last)->next=p0;
		(*last)=p0;
	}
}
//链表释放
void releaselist(rectloc *head)
{
	rectloc *p;
	while(head)
	{
		p=head;
		head=head->next;
		free(p);
	}

}

//链表打印
void listprint(rectloc *head)
{
	rectloc *p=head;
	while(p)
	{		
		if(p->flage)
			printf("%d  %d  %d  %d \n",p->x,p->y,p->width,p->height);
		p=p->next;
	}
}


bool readrectloc(char *filename,rectloc **head1)
{

	char *axis;
	//char fname[10]="zaz1.txt";

	//rectloc *head=NULL,*p0=NULL,*last=NULL;
	rectloc *p0=NULL,*last=NULL,*head=NULL;
	FILE *fp;
	(*head1)=head;
	if((fp=fopen(filename,"rt"))==NULL)
	{
		printf("Can't open file!\n");
		return false;
	}
	else
		while(!feof(fp))
		{
			char loca[21];
			fgets(loca,20,fp);	
			if(loca[0]=='@')            //如果这一行是“@”，代表数据区结束。			
			{
				break;
			}               
			p0=(rectloc*)malloc(LEN);
			if(loca[0]=='#')            //如果这一行是“#”，代表原图中没有手。
			{
				p0->flage=false;
				p0->x=0;
				p0->y=0;
				p0->width=0;
				p0->height=0;
				p0->next=NULL;
				RectLocatlist(&head,p0,&last); //将这一行插入链表
				(*head1)=head;
			}
			else
			{
				p0->flage=true;

				axis=strtok(loca,"	");
				p0->x=atoi(axis);

				axis=strtok(NULL,"	");
				p0->y=atoi(axis);

				axis=strtok(NULL,"	");
				p0->width=atoi(axis);

				axis=strtok(NULL,"	");
				p0->height=atoi(axis);

				/*axis=strtok(loca," ");
				p0->x=atoi(axis);

				axis=strtok(NULL," ");
				p0->y=atoi(axis);

				axis=strtok(NULL," ");
				p0->width=atoi(axis);

				axis=strtok(NULL," ");
				p0->height=atoi(axis);*/

				p0->next=NULL;

				RectLocatlist(&head,p0,&last); //将这一行插入链表
			}
		}
//	listprint(head);
//	releaselist(head);
		fclose(fp);
	return true;
}

//初始化数据

bool initialdata(struct initstruct *D,char *filename)
{
	rectloc *head1=NULL;

	(*D).handnum=0;
	(*D).totalrect=0;
	(*D).thresh=0.8;
	memset((*D).qjl,0,sizeof(int)*MAXRECT);
	memset((*D).zj_num,0,sizeof(int)*MAXRECT);
	memset((*D).zjl,0,sizeof(float)*MAXRECT);
	(*D).qj=0;
	(*D).zj=0;
	(*D).head=NULL;
	(*D).pt=NULL;
	D->time_sum = 0;
	bool fryn;
	fryn=readrectloc(filename,&head1);
	if(fryn==false)
	{
		printf("Can't open file!\n");
		return false;
	}
	(*D).head=head1;
	(*D).pt=head1;
	(*D).tempfilez=fopen("zjl.txt","at+");
	(*D).tempfileq=fopen("qjl.txt","at+");
	return true;
	}

//求是否相交

int	getintersect(struct initstruct *D,rectloc **pt,int result,int h,CvSeq *pOutput, double time)
{
		CvRect rect0,rect1;
		rectloc *p;
		bool yon;
		int num=0;

		if((*pt)==NULL)
			return 0;    //数据完
		p=(*pt);
		(*pt)=(**pt).next;

		if((*p).flage==true)
			(*D).handnum++;

		rect0.x=(*p).x;
		rect0.y=(*p).y;
		rect0.width=(*p).width;
		rect0.height=(*p).height;

		int j=0;

		for(int i=0;i<result;i++)
		{
			(*D).totalrect++;
			CvRect* r = (CvRect*)cvGetSeqElem( pOutput, i );
			rect1.x=r->x;
			rect1.y=r->y;
			rect1.width=r->width;
			rect1.height=r->height;

			yon=rectarea(rect0,rect1,(*D).thresh);  //返回是否框到手
			if(yon)
			{
				(*D).qjl[h-1]=1;
				num++;
			
			}
		}

		fprintf((*D).tempfileq,"%d\n",(*D).qjl[h-1]);

		printf("qjyn = %d   ",(*D).qjl[h-1]);
	
		if(result<=0) {
			(*D).zjl[h-1]=0;
			D->zj_num[h-1]=0;
		} else {
			(*D).zjl[h-1]=(num*1.0)/result;
			D->zj_num[h-1]=result;
		}
		D->time_sum += time;
		fprintf((*D).tempfilez,"%d   %d   %f\n",num,result,(num*1.0)/result);
		printf("num = %d   result = %d   rate = %f\n",num,result,(num*1.0)/result);
		num=0;
		return 1;      //正常结束
}

//结束处理
void endaddest(struct initstruct *D,int h)
{

		releaselist((*D).head);  //释放链表，计算全检率，准检率
		float sumq=0,sumz=0;
		int wj_num=0; /* for the false positive target that is detect */
		for(int i=0;i<h;i++)
		{
			sumq+=(*D).qjl[i];
			sumz+=(*D).zjl[i];
			wj_num += D->zj_num[i] - D->qjl[i];
			//printf("  %d     %f\n", qjl[i],zjl[i]);
		}
		(*D).qj=(1.0*sumq)/(*D).handnum;
		(*D).zj=sumz/h;
		double average_time = 0;
		printf("sumq = %f   sumz = %f    handnum = %d  wujian = %d  h = %d  \n", 
				sumq,sumz,(*D).handnum,wj_num,h);
		printf("qj = %f   zj = %f\n", (*D).qj,(*D).zj);
		//printf("total time = %lf   average time = %lf\n", (*D).time_sum,(*D).time_sum/h);

		fprintf((*D).tempfileq,"#######################################\n");
		fprintf((*D).tempfileq,"qjl=%f\n",(*D).qj);
		fprintf((*D).tempfilez,"#######################################\n");
		fprintf((*D).tempfilez,"zjl=%f\n",(*D).zj);
		fprintf((*D).tempfilez,"zj num = %d\n",wj_num);
		fclose((*D).tempfileq);
		fclose((*D).tempfilez);
}