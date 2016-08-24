// diannaochuquanDlg.h : ͷ�ļ�
//

#pragma once
#include <Windows.h>
#include "resource.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <atlimage.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "afxwin.h"
#include "afxcmn.h"
#include <atltime.h>

using namespace cv;
void MatToCImage(Mat& mat, CImage& cimage);

UINT ThreadFunc_COM(LPVOID pParm); //�̺߳�������
UINT ThreadFunc_HUM(LPVOID pParm);
UINT ThreadFunc_TIME(LPVOID pParm);

struct threadInfo_COM
{
	int p_count;
	int p_list;
	HWND hWnd; //�����ھ����������Ϣ����
};

struct threadInfo_HUM 
{
	HWND hWnd;
};

struct threadInfo_TIME
{
	int time;
	HWND hWnd;
};

// CdiannaochuquanDlg �Ի���
class CdiannaochuquanDlg : public CDialog
{
// ����
public:
	CdiannaochuquanDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_DIANNAOCHUQUAN_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


// ʵ��
protected:
	HICON m_hIcon;
	CWinThread *m_pThread;
	CWinThread *m_hThread;
	CWinThread *m_tThread;
	threadInfo_COM m_Info;
	threadInfo_HUM h_Info;
	threadInfo_TIME t_Info;

	// ���ɵ���Ϣӳ�亯��
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:
	Mat image;

	//list value
	CListBox list_control;
	int list_line;
	afx_msg void OnBnClickedButton1();

	//time value
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	int edit_2_value;

	//display image
	LRESULT DisplayComputer(WPARAM wParam, LPARAM lParam);

	//display video
	LRESULT DisplayVideo(WPARAM wParam, LPARAM lParam);
	afx_msg void OnBnClickedButton2();
	int edit_3_value;
	CString edit_3_str;
	bool flag;
};
