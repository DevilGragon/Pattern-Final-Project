// diannaochuquanDlg.cpp : ʵ���ļ�
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


#define CAMERA			//����ͷ����

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

// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// �Ի�������
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
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


// CdiannaochuquanDlg �Ի���




CdiannaochuquanDlg::CdiannaochuquanDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CdiannaochuquanDlg::IDD, pParent)
	, edit_2_value(0)
	, edit_3_str(_T(""))
{ //���ݳ�ʼ������������
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


// CdiannaochuquanDlg ��Ϣ�������

BOOL CdiannaochuquanDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
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

	// ���ô˶Ի����ͼ�ꡣ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO: �ڴ���Ӷ���ĳ�ʼ������

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
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

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CdiannaochuquanDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
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
	//�ؽ�cimage  
	cimage.Destroy();  
	cimage.Create(nWidth, nHeight, 8 * nChannels);  
	//��������  
	uchar* pucRow;                                  //ָ������������ָ��  
	uchar* pucImage = (uchar*)cimage.GetBits();     //ָ����������ָ��  
	int nStep = cimage.GetPitch();                  //ÿ�е��ֽ���,ע���������ֵ�����и�  
	if (1 == nChannels)                             //���ڵ�ͨ����ͼ����Ҫ��ʼ����ɫ��  
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
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	edit_2_value = 5;
	SetTimer(TIMERID1, 1000, 0);
}

void CdiannaochuquanDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: �ڴ������Ϣ�����������/�����Ĭ��ֵ
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
		AfxMessageBox("�߳�����ʧ�ܣ�",MB_OK|MB_ICONERROR);
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
		CString display = "ʯͷ";
		CString temp;
		CString win_result;
		if(hum_label == 0)
		{
			win_result = "ƽ�֣�";
		}
		else if(hum_label == 1)
		{
			win_result = "��Ӯ��";
		}
		else if(hum_label == 2)
		{
			win_result = "���䣡";
		}
		else
		{
			win_result = "�ز£�";
		}
		temp.Format("%d",lParam);
		list_control.InsertString(-1, temp + "." + display + " " + win_result);
		image = cv::imread("1.png");
		CImage img;
		MatToCImage(image, img);

		CRect rect;//���������   
		int cx = img.GetWidth();//��ȡͼƬ���   
		int cy = img.GetHeight();//��ȡͼƬ�߶�   
		GetDlgItem(IDC_PICTURE1)->GetWindowRect(&rect);//�����ھ���ѡ�е�picture�ؼ���   
		ScreenToClient(&rect);//���ͻ���ѡ�е�Picture�ؼ���ʾ�ľ���������   
		GetDlgItem(IDC_PICTURE1)->MoveWindow(rect.left, rect.top, cx, cy, TRUE);//�������ƶ���Picture�ؼ���ʾ�ľ�������   
		CWnd *pWnd=GetDlgItem(IDC_PICTURE1);//���pictrue�ؼ����ڵľ��   
		pWnd->GetClientRect(&rect);//���pictrue�ؼ����ڵľ�������   
		CDC *pDC=pWnd->GetDC();//���pictrue�ؼ���DC   
		img.Draw(pDC->m_hDC, rect); //��ͼƬ����Picture�ؼ���ʾ�ľ�������   
		ReleaseDC(pDC);//�ͷ�picture�ؼ���DC
	}
	else if (dis_num == 2)
	{
		CString display = "����";
		CString temp;
		CString win_result;
		if(hum_label == 0)
		{
			win_result = "��Ӯ��";
		}
		else if(hum_label == 1)
		{
			win_result = "���䣡";
		}
		else if(hum_label == 2)
		{
			win_result = "ƽ�֣�";
		}
		else
		{
			win_result = "�ز£�";
		}
		temp.Format("%d",lParam);
		list_control.InsertString(-1, temp + "." + display + " " + win_result);
		image = cv::imread("2.png");
		CImage img;
		MatToCImage(image, img);
		CRect rect;//���������   
		int cx = img.GetWidth();//��ȡͼƬ���   
		int cy = img.GetHeight();//��ȡͼƬ�߶�   
		GetDlgItem(IDC_PICTURE1)->GetWindowRect(&rect);//�����ھ���ѡ�е�picture�ؼ���   
		ScreenToClient(&rect);//���ͻ���ѡ�е�Picture�ؼ���ʾ�ľ���������   
		GetDlgItem(IDC_PICTURE1)->MoveWindow(rect.left, rect.top, cx, cy, TRUE);//�������ƶ���Picture�ؼ���ʾ�ľ�������   
		CWnd *pWnd=GetDlgItem(IDC_PICTURE1);//���pictrue�ؼ����ڵľ��   
		pWnd->GetClientRect(&rect);//���pictrue�ؼ����ڵľ�������   
		CDC *pDC=pWnd->GetDC();//���pictrue�ؼ���DC   
		img.Draw(pDC->m_hDC, rect); //��ͼƬ����Picture�ؼ���ʾ�ľ�������   
		ReleaseDC(pDC);//�ͷ�picture�ؼ���DC
	}
	else if (dis_num == 3)
	{
		CString display = "��";
		CString temp;
		CString win_result;
		if(hum_label == 0)
		{
			win_result = "���䣡";
		}
		else if(hum_label == 1)
		{
			win_result = "ƽ�֣�";
		}
		else if(hum_label == 2)
		{
			win_result = "��Ӯ��";
		}
		else
		{
			win_result = "�ز£�";
		}
		temp.Format("%d",lParam);
		list_control.InsertString(-1, temp + "." + display + "    " + win_result);
		image = cv::imread("3.png");
		CImage img;
		MatToCImage(image, img);
		CRect rect;	//���������
		int cx = img.GetWidth();//��ȡͼƬ���   
		int cy = img.GetHeight();//��ȡͼƬ�߶�   
		GetDlgItem(IDC_PICTURE1)->GetWindowRect(&rect);//�����ھ���ѡ�е�picture�ؼ���   
		ScreenToClient(&rect);//���ͻ���ѡ�е�Picture�ؼ���ʾ�ľ���������   
		GetDlgItem(IDC_PICTURE1)->MoveWindow(rect.left, rect.top, cx, cy, TRUE);//�������ƶ���Picture�ؼ���ʾ�ľ�������   
		CWnd *pWnd=GetDlgItem(IDC_PICTURE1);//���pictrue�ؼ����ڵľ��   
		pWnd->GetClientRect(&rect);//���pictrue�ؼ����ڵľ�������   
		CDC *pDC=pWnd->GetDC();//���pictrue�ؼ���DC   
		img.Draw(pDC->m_hDC, rect); //��ͼƬ����Picture�ؼ���ʾ�ľ�������   
		ReleaseDC(pDC);//�ͷ�picture�ؼ���DC
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
	sprintf(filename, "E:\\����\\���ϵͳ�������������\\����������δ���\\fist\\%d.bmp",filecount++);//"F:\\����\\NUS Hand Posture dataset-II\\Hand Postures with human noise\\f_HN (%d).jpg", filecount++);
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
							edit_3_str = "ʯͷ";
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
							edit_3_str = "��";
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
							edit_3_str = "����";
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
				edit_3_str = "���³�ȭ";
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
		sprintf(filename,"E:\\����\\���ϵͳ�������������\\����������δ���\\fist\\%d.bmp",filecount++);//"F:\\����\\NUS Hand Posture dataset-II\\Hand Postures with human noise\\f_HN (%d).jpg" , filecount++);
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
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	edit_3_value = 60;
	flag = 0;
	hum_label = 5;
	h_Info.hWnd = m_hWnd;
	m_hThread = AfxBeginThread(ThreadFunc_HUM, &h_Info);
}
