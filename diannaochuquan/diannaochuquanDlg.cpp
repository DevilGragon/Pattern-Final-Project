// diannaochuquanDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "objdetect1.hpp"
#include "diannaochuquan.h"
#include "diannaochuquanDlg.h"

#include "precomp.hpp"
#include <iostream>
#include <fstream>
#include "stdio.h"
#include <opencv2/opencv.hpp>
#include "rectloarea.h"
#include <string.h>


using namespace cv;
using namespace std;

extern void initialize_lookup_table();
void SkinCrCbDetect(IplImage* src,IplImage* dst);
void CandidateSkinArea(IplImage* src,CvMemStorage* store,vector<Rect>& candidate_SkinArea);
void ChangeArea( CvRect& candidateRect1, IplImage* image);
void classifier_hand( Mat& image,  map<string, CvSVM*>&  svms_map, int& category );
void LoadSvms(const string& svms_dir, map<string, CvSVM*>&  svms_map);


#define CAMERA			//摄像头输入

#ifndef MAIN_PATTERN_COLOR_H
#define MAIN_PATTERN_COLOR_H
static CvScalar colors[] =
{
	{{0,0,255}},
	{{0,128,255}},
	{{0,255,255}},
	{{0,255,0}},
	{{255,128,0}},
	{{255,255,0}},
	{{255,0,0}},
	{{255,0,255}}
};

char divide1[3]={'a','d','h'};
char divide2[3]={'c','e','g'};
char divide3[3]={'f','i','j'};

int hum_label = 5;
#endif

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//DWORD WINAPI computer_view(LPVOID lpParameter);
//DWORD WINAPI human_view(LPVOID lpParameter);

// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CdiannaochuquanDlg 对话框




CdiannaochuquanDlg::CdiannaochuquanDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CdiannaochuquanDlg::IDD, pParent)
	, edit_2_value(0)
	, edit_3_str(_T(""))
{ //数据初始化都放在这里
	list_line = 1;
	m_hIcon = AfxGetApp()->LoadIcon(IDI_MAINFRAME);
	m_pThread = NULL;
	m_hThread = NULL;
}

void CdiannaochuquanDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LIST1, list_control);
	DDX_Text(pDX, IDC_EDIT2, edit_2_value);
	DDX_Text(pDX, IDC_EDIT3, edit_3_str);
}

BEGIN_MESSAGE_MAP(CdiannaochuquanDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CdiannaochuquanDlg::OnBnClickedButton1)
	ON_WM_TIMER()
	ON_MESSAGE(WM_INFO1, &CdiannaochuquanDlg::DisplayComputer)
	ON_MESSAGE(WM_INFO2, &CdiannaochuquanDlg::DisplayVideo)
	ON_BN_CLICKED(IDC_BUTTON2, &CdiannaochuquanDlg::OnBnClickedButton2)
END_MESSAGE_MAP()


// CdiannaochuquanDlg 消息处理程序

BOOL CdiannaochuquanDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CdiannaochuquanDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CdiannaochuquanDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CdiannaochuquanDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void MatToCImage(Mat& mat, CImage& cimage)  
{  
	if (0 == mat.total())  
	{  
		return;  
	}  
	int nChannels = mat.channels();  
	if ((1 != nChannels) && (3 != nChannels))  
	{  
		return;  
	}  
	int nWidth    = mat.cols;  
	int nHeight   = mat.rows;  
	//重建cimage  
	cimage.Destroy();  
	cimage.Create(nWidth, nHeight, 8 * nChannels);  
	//拷贝数据  
	uchar* pucRow;                                  //指向数据区的行指针  
	uchar* pucImage = (uchar*)cimage.GetBits();     //指向数据区的指针  
	int nStep = cimage.GetPitch();                  //每行的字节数,注意这个返回值有正有负  
	if (1 == nChannels)                             //对于单通道的图像需要初始化调色板  
	{  
		RGBQUAD* rgbquadColorTable;  
		int nMaxColors = 256;  
		rgbquadColorTable = new RGBQUAD[nMaxColors];  
		cimage.GetColorTable(0, nMaxColors, rgbquadColorTable);  
		for (int nColor = 0; nColor < nMaxColors; nColor++)  
		{  
			rgbquadColorTable[nColor].rgbBlue = (uchar)nColor;  
			rgbquadColorTable[nColor].rgbGreen = (uchar)nColor;  
			rgbquadColorTable[nColor].rgbRed = (uchar)nColor;  
		}  
		cimage.SetColorTable(0, nMaxColors, rgbquadColorTable);  
		delete []rgbquadColorTable;  
	}  
	for (int nRow = 0; nRow < nHeight; nRow++)  
	{  
		pucRow = (mat.ptr<uchar>(nRow));  
		for (int nCol = 0; nCol < nWidth; nCol++)  
		{  
			if (1 == nChannels)  
			{  
				*(pucImage + nRow * nStep + nCol) = pucRow[nCol];  
			}  
			else if (3 == nChannels)  
			{  
				for (int nCha = 0 ; nCha < 3; nCha++)  
				{  
					*(pucImage + nRow * nStep + nCol * 3 + nCha) = pucRow[nCol * 3 + nCha];  
				}             
			}  
		}     
	}
}

void CdiannaochuquanDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	edit_2_value = 5;
	SetTimer(TIMERID1, 1000, 0);
}

void CdiannaochuquanDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	CString str;
	str.Format("%d", edit_2_value);
	edit_2_value--;
	SetDlgItemText(IDC_EDIT2, str);

	m_Info.p_list = list_line;
	m_Info.p_count = edit_2_value;
	m_Info.hWnd = m_hWnd;

	m_pThread = AfxBeginThread(ThreadFunc_COM, &m_Info);

	if(m_pThread == NULL)
	{
		AfxMessageBox("线程启动失败！",MB_OK|MB_ICONERROR);
		exit(0);
	}
	CDialog::OnTimer(nIDEvent);
}

UINT ThreadFunc_COM(LPVOID pParm)
{
	threadInfo_COM *pInfo=(threadInfo_COM*)pParm;
	int p_edit_2_value = pInfo->p_count;
	int p_list_line = pInfo->p_list;
	if(p_edit_2_value < 0)
	{
		srand((unsigned)time(NULL));
		int num = 0;
		num = rand() % 3 + 1;
		::SendMessage(pInfo->hWnd, WM_INFO1, num, p_list_line);
	}
	return 0;
}

LRESULT CdiannaochuquanDlg::DisplayComputer(WPARAM wParam, LPARAM lParam)
{
	int dis_num;
	dis_num = wParam; //1 2 3
	if (dis_num == 1)
	{
		CString display = "石头";
		CString temp;
		CString win_result;
		if(hum_label == 0)
		{
			win_result = "平手！";
		}
		else if(hum_label == 1)
		{
			win_result = "人赢！";
		}
		else if(hum_label == 2)
		{
			win_result = "人输！";
		}
		else
		{
			win_result = "重猜！";
		}
		temp.Format("%d",lParam);
		list_control.InsertString(-1, temp + "." + display + " " + win_result);
		image = cv::imread("1.png");
		CImage img;
		MatToCImage(image, img);

		CRect rect;//定义矩形类   
		int cx = img.GetWidth();//获取图片宽度   
		int cy = img.GetHeight();//获取图片高度   
		GetDlgItem(IDC_PICTURE1)->GetWindowRect(&rect);//将窗口矩形选中到picture控件上   
		ScreenToClient(&rect);//将客户区选中到Picture控件表示的矩形区域内   
		GetDlgItem(IDC_PICTURE1)->MoveWindow(rect.left, rect.top, cx, cy, TRUE);//将窗口移动到Picture控件表示的矩形区域   
		CWnd *pWnd=GetDlgItem(IDC_PICTURE1);//获得pictrue控件窗口的句柄   
		pWnd->GetClientRect(&rect);//获得pictrue控件所在的矩形区域   
		CDC *pDC=pWnd->GetDC();//获得pictrue控件的DC   
		img.Draw(pDC->m_hDC, rect); //将图片画到Picture控件表示的矩形区域   
		ReleaseDC(pDC);//释放picture控件的DC
	}
	else if (dis_num == 2)
	{
		CString display = "剪刀";
		CString temp;
		CString win_result;
		if(hum_label == 0)
		{
			win_result = "人赢！";
		}
		else if(hum_label == 1)
		{
			win_result = "人输！";
		}
		else if(hum_label == 2)
		{
			win_result = "平手！";
		}
		else
		{
			win_result = "重猜！";
		}
		temp.Format("%d",lParam);
		list_control.InsertString(-1, temp + "." + display + " " + win_result);
		image = cv::imread("2.png");
		CImage img;
		MatToCImage(image, img);
		CRect rect;//定义矩形类   
		int cx = img.GetWidth();//获取图片宽度   
		int cy = img.GetHeight();//获取图片高度   
		GetDlgItem(IDC_PICTURE1)->GetWindowRect(&rect);//将窗口矩形选中到picture控件上   
		ScreenToClient(&rect);//将客户区选中到Picture控件表示的矩形区域内   
		GetDlgItem(IDC_PICTURE1)->MoveWindow(rect.left, rect.top, cx, cy, TRUE);//将窗口移动到Picture控件表示的矩形区域   
		CWnd *pWnd=GetDlgItem(IDC_PICTURE1);//获得pictrue控件窗口的句柄   
		pWnd->GetClientRect(&rect);//获得pictrue控件所在的矩形区域   
		CDC *pDC=pWnd->GetDC();//获得pictrue控件的DC   
		img.Draw(pDC->m_hDC, rect); //将图片画到Picture控件表示的矩形区域   
		ReleaseDC(pDC);//释放picture控件的DC
	}
	else if (dis_num == 3)
	{
		CString display = "布";
		CString temp;
		CString win_result;
		if(hum_label == 0)
		{
			win_result = "人输！";
		}
		else if(hum_label == 1)
		{
			win_result = "平手！";
		}
		else if(hum_label == 2)
		{
			win_result = "人赢！";
		}
		else
		{
			win_result = "重猜！";
		}
		temp.Format("%d",lParam);
		list_control.InsertString(-1, temp + "." + display + "    " + win_result);
		image = cv::imread("3.png");
		CImage img;
		MatToCImage(image, img);
		CRect rect;	//定义矩形类
		int cx = img.GetWidth();//获取图片宽度   
		int cy = img.GetHeight();//获取图片高度   
		GetDlgItem(IDC_PICTURE1)->GetWindowRect(&rect);//将窗口矩形选中到picture控件上   
		ScreenToClient(&rect);//将客户区选中到Picture控件表示的矩形区域内   
		GetDlgItem(IDC_PICTURE1)->MoveWindow(rect.left, rect.top, cx, cy, TRUE);//将窗口移动到Picture控件表示的矩形区域   
		CWnd *pWnd=GetDlgItem(IDC_PICTURE1);//获得pictrue控件窗口的句柄   
		pWnd->GetClientRect(&rect);//获得pictrue控件所在的矩形区域   
		CDC *pDC=pWnd->GetDC();//获得pictrue控件的DC   
		img.Draw(pDC->m_hDC, rect); //将图片画到Picture控件表示的矩形区域   
		ReleaseDC(pDC);//释放picture控件的DC
	}
	list_line++;
	KillTimer(TIMERID1);
	return 0;
}

UINT ThreadFunc_HUM(LPVOID pParm)
{
	threadInfo_HUM *hInfo=(threadInfo_HUM*)pParm;
	::SendMessage(hInfo->hWnd,WM_INFO2, 1, 1);
	return 0;
}

LRESULT CdiannaochuquanDlg::DisplayVideo(WPARAM wParam, LPARAM lParam)
{
	char time[2];
	string edit_3_value_str;
	CvFont font;    
	double hScale=1;   
	double vScale=1;    
	int lineWidth=2;
	cvInitFont(&font,CV_FONT_HERSHEY_DUPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);

	initialize_lookup_table();
	CvCapture* capture = 0;

#ifdef CAMERA
	capture = cvCaptureFromCAM(0);
	if( !capture )
	{
		cout<<"err in cvCaptureFromCAM"<<endl;
		system("PAUSE");
		return 0;
	}
#endif

#ifdef VIDEO
	capture = cvCreateFileCapture(input_name);
	if (capture == NULL) {
		printf("failed to get capture from %s\n", input_name);
		return -1;
	}
#endif

#ifdef DOCUMENT
	char filename[500];
	int filecount = 0;
#endif

#ifdef SAVE_NEG
	int fault_count=0;
	char * fault_name = (char *)calloc(20,sizeof(char));

	IplImage *sub_image, *copy_image;
#endif

#ifdef TEST
	int frame_count = 0;
	char * frame_name = (char *)calloc(20,sizeof(char));
#endif

	int width  = 40;
	int height = 40;

	char* classifierdir= NULL;

	classifierdir="..\\classifier\\classifier_result_mysample";

	CvMiscClassifierCascade* cascade = cvLoadMiscClassifierCascade( classifierdir, cvSize( width, height ) );

#if 0
	string svms_dir = "svms/";
#else
	string svms_dir = "..\\gaor\\dense_sift_svms_gaor";
#endif
	map<string, CvSVM*>  svms_map;    
	LoadSvms(svms_dir, svms_map);

	if( !cascade )
	{
		cout<<"err in cvLoad"<<endl;
		system("PAUSE");
		return 0;
	}

	IplImage *capture_image, *gray, *src_image;
#ifdef DOCUMENT
	sprintf(filename, "E:\\毕设\\检测系统测试样本及结果\\测试样本—未标记\\fist\\%d.bmp",filecount++);//"F:\\样本\\NUS Hand Posture dataset-II\\Hand Postures with human noise\\f_HN (%d).jpg", filecount++);
	capture_image = cvLoadImage( filename, CV_LOAD_IMAGE_COLOR);
	gray = cvCreateImage(cvSize(capture_image->width, capture_image->height), 8, 0);
	IplImage* src_temp = cvCreateImage(cvSize(capture_image->width, capture_image->height), \
		capture_image->depth, capture_image->nChannels);
	IplImage* dst_temp=cvCreateImage(cvGetSize(src_temp),src_temp->depth,1);

#else
	src_image = cvQueryFrame( capture );
	CvSize sz; 
	double width_scale = 2;
	double height_scale = 1.5;
	sz.width = (int)(src_image->width/width_scale);
	sz.height = (int)(src_image->height/height_scale);
	capture_image = cvCreateImage(sz,src_image->depth,src_image->nChannels);
	cvResize(src_image,capture_image,CV_INTER_CUBIC); 
	gray = cvCreateImage(cvSize(capture_image->width/2, capture_image->height/2), 8, 1);
	IplImage* src_temp = cvCreateImage(cvSize(capture_image->width/2, capture_image->height/2), \
		capture_image->depth,capture_image->nChannels);
	IplImage* dst_temp=cvCreateImage(cvGetSize(src_temp),src_temp->depth,1);
#endif

#ifdef SAVE_NEG
	copy_image = cvCreateImage(cvSize(capture_image->width, capture_image->height), 8, 3);
#endif

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvMemStorage* store=cvCreateMemStorage();
	cvClearMemStorage(storage);
	cvClearMemStorage(store);
	int resize_pic = CV_HAAR_SCALE_IMAGE;

	CvSeq* faces;
	double t = 0.0;
	int file_count=0;
	char * file_name = (char *)calloc(20,sizeof(char));

	vector<Rect> candidate_SkinArea;

	FILE* timelist;
	timelist = fopen("timelist.txt","w");

	cascade->count = 20;	
	while(capture)
	{
		edit_3_value--;
		t = (double)cvGetTickCount();
#ifdef DOCUMENT	
		cvCopy(capture_image, src_temp);
		cvCvtColor(src_temp, gray, CV_BGR2GRAY);

#else 
		cvResize(capture_image,src_temp);
		cvCvtColor(src_temp, gray, CV_BGR2GRAY);

#endif
		SkinCrCbDetect(src_temp, dst_temp);
		candidate_SkinArea.clear();
		CandidateSkinArea(dst_temp, store, candidate_SkinArea);
		int lable;
		string type;
		int num;

		for (int i = 0; i < (int)candidate_SkinArea.size(); i++)
		{
			num = 0;
			CvRect SkinRect = candidate_SkinArea[i];
			cvSetImageROI(gray, SkinRect);
			faces = cvMiscDetectObjects( gray, cascade, storage, 1.1, 3,\
				resize_pic, cvSize(40,40) );
			cvResetImageROI(gray);
			if(faces!=NULL)
				num = num+faces->total;
			for(int j = 0; j < (faces ? faces->total : 0);j++)
			{
				CvRect* r = (CvRect*)cvGetSeqElem( faces, 0 );
				CvRect detectArea = cvRect((SkinRect.x + r->x),(SkinRect.y + r->y),\
					r->width, r->height);

				cvRectangle(src_temp,cvPoint(detectArea.x,detectArea.y),\
					cvPoint((detectArea.x+detectArea.width),(detectArea.y+detectArea.height)),colors[i],3,8,0);
				cvSetImageROI(gray,detectArea);
				Mat tempimage(gray,0);
				resize(tempimage, tempimage, Size(60, 60));
				classifier_hand(tempimage, svms_map, lable);
				printf("label is ++++++++++%d\n",lable);

				switch ( lable )
				{
					case 0:
					{
						type = "fist";
						if((edit_3_value > 0) && (edit_3_value < 20))
						{
							hum_label = lable;
							edit_3_str = "石头";
							flag = 1;
						}
						break;
					}
					case 1:
					{
						type = "palm"; 
						if((edit_3_value > 0) && (edit_3_value < 20))
						{
							hum_label = lable;
							edit_3_str = "布";
							flag = 1;
						}
						break;
					}
					case 2:
					{
						type = "victor";
						if((edit_3_value > 0) && (edit_3_value < 20))
						{
							hum_label = lable;
							edit_3_str = "剪刀";
							flag = 1;
						}
						break;
					}
				}
				const char *p=type.c_str();
				cvPutText(src_temp, p,cvPoint(25,25),&font,CV_RGB(255,0,0) );
				cvResetImageROI(gray);
				/*SetDlgItemText(IDC_EDIT3, edit_3_str);*/

			}
			itoa((edit_3_value / 10), time, 10);
			edit_3_value_str = time;
			const char *count = edit_3_value_str.c_str();
			if((edit_3_value / 10 == 0)	|| (edit_3_value / 10 == 1) || (edit_3_value / 10 == 2)\
				|| (edit_3_value / 10 == 3) || (edit_3_value / 10 == 4) || (edit_3_value / 10 == 5))
			{
				cvPutText(src_temp, count,cvPoint(275,25),&font,CV_RGB(124,252,0) );
			}
			if(flag == 0)
				edit_3_str = "重新出拳";
			SetDlgItemText(IDC_EDIT3, edit_3_str);
		}
		t = (double)cvGetTickCount() - t;
		double tt = (float)t/((double)cvGetTickFrequency()*1000.);
		fprintf(timelist,"%f\n",tt);
		fflush(stdout);

#ifdef SAVE_NEG
		cvCopy(capture_image,copy_image);
#endif


#ifdef DOCUMENT
		cvShowImage( "result", src_temp);
#else
		CImage img_result;
		Mat src_temp_mat = src_temp;
		MatToCImage(src_temp_mat, img_result);

		CRect rect2;  
		int cx2 = img_result.GetWidth();   
		int cy2 = img_result.GetHeight();   
		GetDlgItem(IDC_PICTURE2)->GetWindowRect(&rect2);
		ScreenToClient(&rect2);  
		GetDlgItem(IDC_PICTURE2)->MoveWindow(rect2.left, rect2.top, cx2, cy2, TRUE);   
		CWnd *pWnd2=GetDlgItem(IDC_PICTURE2);
		pWnd2->GetClientRect(&rect2);
		CDC *pDC2=pWnd2->GetDC(); 
		img_result.Draw(pDC2->m_hDC, rect2);   
		ReleaseDC(pDC2);
#endif
		int c = cvWaitKey(-1);
		if (c == 27 || c == 'q')
			break;

#ifdef TEST
		sprintf(frame_name, "..\\mysample_result\\group2\\%d.jpg",frame_count);//"..\\mysample_result\\f_HN (%d).jpg",frame_count);
		cvSaveImage(frame_name, src_temp);
		frame_count++;
#endif	

#ifdef DOCUMENT
		sprintf(filename,"E:\\毕设\\检测系统测试样本及结果\\测试样本—未标记\\fist\\%d.bmp",filecount++);//"F:\\样本\\NUS Hand Posture dataset-II\\Hand Postures with human noise\\f_HN (%d).jpg" , filecount++);
		//printf("%s\n", filename);
		capture_image = cvLoadImage( filename, CV_LOAD_IMAGE_COLOR);

#else
		capture_image = cvQueryFrame( capture );
#endif
		if(edit_3_value <= 0)
		{
			goto label_1;
		}
	}

#ifdef SAVE_NEG
	cvReleaseImage(&copy_image);
#endif

label_1:
	cvReleaseImage(&capture_image);
	cvReleaseImage(&gray);
	cvReleaseImage(&src_temp);

	cvReleaseMemStorage(&storage);
	cvReleaseMemStorage(&store);

	fclose(timelist);
	return 0;
}

void CdiannaochuquanDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	edit_3_value = 60;
	flag = 0;
	hum_label = 5;
	h_Info.hWnd = m_hWnd;
	m_hThread = AfxBeginThread(ThreadFunc_HUM, &h_Info);
}
