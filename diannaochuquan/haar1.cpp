/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* Misc features calculation */
#include "stdafx.h"

#include "precomp.hpp"
#include <stdio.h>
#include <omp.h> 
/*#if CV_SSE2
#   if CV_SSE4 || defined __SSE4__
#       include <smmintrin.h>
#   else
#       define _mm_blendv_pd(a, b, m) _mm_xor_pd(a, _mm_and_pd(_mm_xor_pd(b, a), m))
#       define _mm_blendv_ps(a, b, m) _mm_xor_ps(a, _mm_and_ps(_mm_xor_ps(b, a), m))
#   endif
#if defined CV_ICC
#   define CV_HAAR_USE_SSE 1
#endif
#endif*/

/* these settings affect the quality of detection: change with care */
#define CV_ADJUST_FEATURES 1
#define CV_ADJUST_WEIGHTS  0

typedef int sumtype;
typedef double sqsumtype;
typedef float  hogsumtype;

extern void initialize_mapbuf(int w, int h);
extern void tHOGIntegral(const CvMat* src, CvMat* grad, int binNum);

typedef struct CvHidMiscHaarFeature{
	struct
	{
		sumtype* p0, *p1, *p2, *p3;
		float weight;
	} rect[CV_HAAR_FEATURE_MAX];
}CvHidMiscHaarFeature;

typedef struct CvHidMiscVarFeature{
	int used_rects;//it can be 1 or 2. if being 1, this feature only calculate the var of r0; if being 2, this feature is calculated as:var(r0)+var(r1)+abs( var(r0) - var(r1))
	struct
	{
		sumtype* p0, *p1, *p2, *p3;
		sqsumtype* pq0,*pq1,*pq2,*pq3;
		int num_pixels;
	} rect[2];
}CvHidMiscVarFeature;


typedef struct CvHidMiscHogFeature{
	struct
	{
		sumtype p0, p1, p2, p3;//根据所在cell位置转化而来		
	}fastRect;

	int inIdx;//[0~8]
}CvHidMiscHogFeature;


typedef struct CvHidMiscFeature
{
	enum MiscFeatureType type;
	union{
		CvHidMiscHaarFeature haar;
		CvHidMiscVarFeature var; 
		CvHidMiscHogFeature hog; 
	}misc;
}CvHidMiscFeature;


typedef struct CvHidMiscClassifier
{
	CvHidMiscFeature features;
	float threshold[MAX_GROUPS];
	float left[MAX_GROUPS];
	float right[MAX_GROUPS];
}
CvHidMiscClassifier;
typedef struct CvHidMiscStageClassifier
{
	int  count;
	float threshold;
	CvHidMiscClassifier* classifier;
	int index[MAX_GROUPS];
	int* group[MAX_GROUPS];
	float subthreshold[MAX_GROUPS];
	struct CvHidMiscStageClassifier* next;
	struct CvHidMiscStageClassifier* child;
	struct CvHidMiscStageClassifier* parent;
}
CvHidMiscStageClassifier;


struct CvHidMiscClassifierCascade
{
	int  count;
	int  isStumpBased;
	//int  has_tilted_features; //we don't use tilted feature.
	int  is_tree;
	double inv_window_area;
	CvMat sum, sqsum,*hogsum;
	CvHidMiscStageClassifier* stage_classifier;
	sqsumtype *pq0, *pq1, *pq2, *pq3;
	sumtype *p0, *p1, *p2, *p3;
	void** ipp_stages;
};


const int icv_object_win_border = 1;
const float icv_stage_threshold_bias = 0.0001f;

static CvMiscClassifierCascade*
icvCreateMiscClassifierCascade( int stage_count )
{
	CvMiscClassifierCascade* cascade = 0;

	int block_size = sizeof(*cascade) + stage_count*sizeof(*cascade->stage_classifier);

	if( stage_count <= 0 )
		CV_Error( CV_StsOutOfRange, "Number of stages should be positive" );

	cascade = (CvMiscClassifierCascade*)cvAlloc( block_size );
	memset( cascade, 0, block_size );

	cascade->stage_classifier = (CvMiscStageClassifier*)(cascade + 1);
	cascade->flags = CV_MISC_MAGIC_VAL;
	cascade->count = stage_count;

	return cascade;
}


static void
icvReleaseHidMiscClassifierCascade( CvHidMiscClassifierCascade** _cascade )
{
	if( _cascade && *_cascade )
	{
#ifdef HAVE_IPP
		CvHidMiscClassifierCascade* cascade = *_cascade;
		if( cascade->ipp_stages )
		{
			int i;
			for( i = 0; i < cascade->count; i++ )
			{
				if( cascade->ipp_stages[i] )
					ippiMiscClassifierFree_32f( (IppiMiscClassifier_32f*)cascade->ipp_stages[i] );
			}
		}
		cvFree( &cascade->ipp_stages );
#endif
		cvFree( _cascade );
	}
}


/* create more efficient internal representation of haar classifier cascade */
static CvHidMiscClassifierCascade*
icvCreateHidMiscClassifierCascade( CvMiscClassifierCascade* cascade )
{
	CvRect* ipp_features = 0;
	float *ipp_weights = 0, *ipp_thresholds = 0, *ipp_val1 = 0, *ipp_val2 = 0;
	int* ipp_counts = 0;

	CvHidMiscClassifierCascade* out = 0;

	int i, j, k;
	int datasize;
	int total_classes = 0;
	int total_indexes=0;
	char errorstr[100];

	CvHidMiscClassifier* misc_classifier_ptr;
	int* misc_classifier_index;
	CvSize orig_window_size;
	int max_count = 0;

	if( !CV_IS_MISC_CLASSIFIER(cascade) )
		CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

	if( cascade->hid_cascade )
		CV_Error( CV_StsError, "hid_cascade has been already created" );

	if( !cascade->stage_classifier )
		CV_Error( CV_StsNullPtr, "" );

	if( cascade->count <= 0 )
		CV_Error( CV_StsOutOfRange, "Negative number of cascade stages" );

	orig_window_size = cascade->orig_window_size;

	/* check input structure correctness and calculate total memory size needed for
	internal representation of the classifier cascade */
	for( i = 0; i < cascade->count; i++ )
	{
		CvMiscStageClassifier* stage_classifier = cascade->stage_classifier + i;

		if( !stage_classifier->classifier ||
			stage_classifier->count <= 0 )
		{
			sprintf( errorstr, "header of the stage classifier #%d is invalid "
				"(has null pointers or non-positive classfier count)", i );
			CV_Error( CV_StsError, errorstr );
		}

		total_classes += stage_classifier->count;
		for(j=0;j<MAX_GROUPS;j++)
		{
			total_indexes+=stage_classifier->group[j].size();
		}

	}

	// this is an upper boundary for the whole hidden cascade size
	datasize = sizeof(CvHidMiscClassifierCascade) +
		sizeof(CvHidMiscStageClassifier)*cascade->count +
		sizeof(CvHidMiscClassifier) * total_classes+
		sizeof(int)*total_indexes;

	out = (CvHidMiscClassifierCascade*)cvAlloc( datasize );
	memset( out, 0, sizeof(*out) );

	/* init header */
	out->count = cascade->count;
	out->stage_classifier = (CvHidMiscStageClassifier*)(out + 1);
	misc_classifier_ptr=(CvHidMiscClassifier*)(out->stage_classifier + cascade->count);
	misc_classifier_index=(int*)(misc_classifier_ptr+total_classes);
	out->isStumpBased = 0;
	out->is_tree = 0;

	/* initialize internal representation */
	for( i = 0; i < cascade->count; i++ )
	{
		CvMiscStageClassifier* stage_classifier = cascade->stage_classifier + i;
		CvHidMiscStageClassifier* hid_stage_classifier = out->stage_classifier + i;

		hid_stage_classifier->count = stage_classifier->count;
		hid_stage_classifier->classifier = misc_classifier_ptr;
		misc_classifier_ptr += stage_classifier->count;

		hid_stage_classifier->parent = (stage_classifier->parent == -1)
			? NULL : out->stage_classifier + stage_classifier->parent;
		hid_stage_classifier->next = (stage_classifier->next == -1)
			? NULL : out->stage_classifier + stage_classifier->next;
		hid_stage_classifier->child = (stage_classifier->child == -1)
			? NULL : out->stage_classifier + stage_classifier->child;

		out->is_tree |= hid_stage_classifier->next != NULL;

		for( j = 0; j < stage_classifier->count; j++ )
		{
			CvMiscClassifier* classifier = stage_classifier->classifier + j;
			CvHidMiscClassifier* hid_classifier = hid_stage_classifier->classifier + j;

			memset( hid_classifier, -1, sizeof(*hid_classifier) );
			for(k=0;k<MAX_GROUPS;k++)
			{
				hid_classifier->left[k]=classifier->left[k];
				hid_classifier->right[k]=classifier->right[k];
				hid_classifier->threshold[k]=classifier->threshold[k];
			}

			CvHidMiscFeature* hidfeature =&( hid_classifier->features);
			CvMiscFeature* feature = &(classifier->misc_feature);
			hidfeature->type = feature->type;

			switch(feature->type){
case MISC_HAAR:
	if( fabs(feature->misc.haar.rect[2].weight) < DBL_EPSILON ||
		feature->misc.haar.rect[2].r.width == 0 ||
		feature->misc.haar.rect[2].r.height == 0 )
		memset( &(hidfeature->misc.haar.rect[2]), 0, sizeof(hidfeature->misc.haar.rect[2]) );
	break;
case MISC_VAR:
	hidfeature->misc.var.used_rects=feature->misc.var.used_rects;
	if( feature->misc.var.used_rects==1 )
	{
		memset( &(hidfeature->misc.var.rect[1]), 0, sizeof(hidfeature->misc.var.rect[1]) );
	}
	break;
case MISC_HOG:
	break;
default:
	assert(0);

			}

		}
		for( j = 0;j < MAX_GROUPS ; j++)
		{
			hid_stage_classifier->index[j]=stage_classifier->group[j].size();
			hid_stage_classifier->subthreshold[j]=stage_classifier->subthreshold[j];
			hid_stage_classifier->group[j]=misc_classifier_index;
			misc_classifier_index+=hid_stage_classifier->index[j];
			for(k=0;k<hid_stage_classifier->index[j];k++)
			{
				hid_stage_classifier->group[j][k]=stage_classifier->group[j][k];
			}
		}
	}


	cascade->hid_cascade = out;

	cvFree( &ipp_features );
	cvFree( &ipp_weights );
	cvFree( &ipp_thresholds );
	cvFree( &ipp_val1 );
	cvFree( &ipp_val2 );
	cvFree( &ipp_counts );

	return out;
}

#define sum_elem_ptr(sum,row,col)  \
	((sumtype*)CV_MAT_ELEM_PTR_FAST((sum),(row),(col),sizeof(sumtype)))

#define sqsum_elem_ptr(sqsum,row,col)  \
	((sqsumtype*)CV_MAT_ELEM_PTR_FAST((sqsum),(row),(col),sizeof(sqsumtype)))

#define calc_sum(rect,offset) \
	((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

#define calc_sqsum(rect,offset) \
	((rect).pq0[offset] - (rect).pq1[offset] - (rect).pq2[offset] + (rect).pq3[offset])



#define CV_SUM_OFFSETS( p0, p1, p2, p3, rect, step )                      \
	/* (x, y) */                                                          \
	(p0) = (rect).x + (step) * (rect).y;                                  \
	/* (x + w, y) */                                                      \
	(p1) = (rect).x + (rect).width + (step) * (rect).y;                   \
	/* (x + w, y) */                                                      \
	(p2) = (rect).x + (step) * ((rect).y + (rect).height);                \
	/* (x + w, y + h) */                                                  \
	(p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);



CV_IMPL void
cvSetImagesForMiscClassifierCascade( CvMiscClassifierCascade* _cascade,
									const CvArr* _sum,
									const CvArr* _sqsum,
									CvMat* _hogsum,
									double scale )
{
	CvMat sum_stub, *sum = (CvMat*)_sum;
	CvMat sqsum_stub, *sqsum = (CvMat*)_sqsum;


	CvHidMiscClassifierCascade* cascade;
	int coi0 = 0, coi1 = 0;
	int i;
	CvRect equRect;
	double weight_scale;



	if( !CV_IS_MISC_CLASSIFIER(_cascade) )
		CV_Error( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

	if( scale <= 0 )
		CV_Error( CV_StsOutOfRange, "Scale must be positive" );

	sum = cvGetMat( sum, &sum_stub, &coi0 );
	sqsum = cvGetMat( sqsum, &sqsum_stub, &coi1 );

	if( coi0 || coi1 )
		CV_Error( CV_BadCOI, "COI is not supported" );

	if( !CV_ARE_SIZES_EQ( sum, sqsum ))
		CV_Error( CV_StsUnmatchedSizes, "All integral images must have the same size" );

	if( CV_MAT_TYPE(sqsum->type) != CV_64FC1 ||
		CV_MAT_TYPE(sum->type) != CV_32SC1 )
		CV_Error( CV_StsUnsupportedFormat,
		"Only (32s, 64f, 32s) combination of (sum,sqsum) formats is allowed" );

	if( !_cascade->hid_cascade )
		icvCreateHidMiscClassifierCascade(_cascade);

	cascade = _cascade->hid_cascade;

	_cascade->scale = scale;//scale = 1
	_cascade->real_window_size.width = cvRound( _cascade->orig_window_size.width * scale );//50
	_cascade->real_window_size.height = cvRound( _cascade->orig_window_size.height * scale );//50

	cascade->sum = *sum;
	cascade->sqsum = *sqsum;
	cascade->hogsum = _hogsum;

	equRect.x = equRect.y =cvRound(scale);//(1,1)
	equRect.width = cvRound((_cascade->orig_window_size.width-2)*scale);//50-2
	equRect.height =cvRound((_cascade->orig_window_size.height-2)*scale);//50-2
	weight_scale = 1./(equRect.width*equRect.height);
	cascade->inv_window_area = equRect.width * equRect.height;

	//为后面窗口移动后计算整个窗口的像素均值提供基础，这里的窗口位置在左上角，(1,1,48,48)
	cascade->p0 = sum_elem_ptr(*sum, equRect.y, equRect.x);
	cascade->p1 = sum_elem_ptr(*sum, equRect.y, equRect.x + equRect.width );
	cascade->p2 = sum_elem_ptr(*sum, equRect.y + equRect.height, equRect.x );
	cascade->p3 = sum_elem_ptr(*sum, equRect.y + equRect.height,
		equRect.x + equRect.width );

	cascade->pq0 = sqsum_elem_ptr(*sqsum, equRect.y, equRect.x);
	cascade->pq1 = sqsum_elem_ptr(*sqsum, equRect.y, equRect.x + equRect.width );
	cascade->pq2 = sqsum_elem_ptr(*sqsum, equRect.y + equRect.height, equRect.x );
	cascade->pq3 = sqsum_elem_ptr(*sqsum, equRect.y + equRect.height,
		equRect.x + equRect.width );



	/* init pointers in haar features according to real window size and
	given image pointers */
	for( i = 0; i < _cascade->count; i++ )
	{
		int j, k, l;
		for( j = 0; j < cascade->stage_classifier[i].count; j++ )
		{
			CvMiscFeature* feature =
				&_cascade->stage_classifier[i].classifier[j].misc_feature;
			CvHidMiscFeature* hidfeature =
				&cascade->stage_classifier[i].classifier[j].features;

			if(hidfeature->type == MISC_HAAR) 
			{              
				CvRect r[3];

				for( int k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
				{
					hidfeature->misc.haar.rect[k].weight = feature->misc.haar.rect[k].weight;

					if( hidfeature->misc.haar.rect[k].weight == 0.0F )
						break;

					r[k] = feature->misc.haar.rect[k].r;
					hidfeature->misc.haar.rect[k].p0 = sum_elem_ptr(*sum, r[k].y, r[k].x);
					hidfeature->misc.haar.rect[k].p1 = sum_elem_ptr(*sum, r[k].y, r[k].x + r[k].width);
					hidfeature->misc.haar.rect[k].p2 = sum_elem_ptr(*sum, r[k].y + r[k].height, r[k].x);
					hidfeature->misc.haar.rect[k].p3 = sum_elem_ptr(*sum, r[k].y + r[k].height, r[k].x + r[k].width);
				}

			}//MISC_HAAR

			else if(hidfeature->type == MISC_VAR){
				CvRect r0 = feature->misc.var.r0;

				hidfeature->misc.var.rect[0].p0 = sum_elem_ptr(*sum,r0.y,r0.x);
				hidfeature->misc.var.rect[0].p1 = sum_elem_ptr(*sum,r0.y,r0.x+r0.width);
				hidfeature->misc.var.rect[0].p2 = sum_elem_ptr(*sum,r0.y+r0.height,r0.x);
				hidfeature->misc.var.rect[0].p3 = sum_elem_ptr(*sum,r0.y+r0.height,r0.x+r0.width); 

				hidfeature->misc.var.rect[0].pq0 = sqsum_elem_ptr(*sqsum,r0.y,r0.x);
				hidfeature->misc.var.rect[0].pq1 = sqsum_elem_ptr(*sqsum,r0.y,r0.x+r0.width);
				hidfeature->misc.var.rect[0].pq2 = sqsum_elem_ptr(*sqsum,r0.y+r0.height,r0.x);
				hidfeature->misc.var.rect[0].pq3 = sqsum_elem_ptr(*sqsum,r0.y+r0.height,r0.x+r0.width);

				hidfeature->misc.var.rect[0].num_pixels = r0.width * r0.height;               	
			}//MISC_VAR
			else if(hidfeature->type == MISC_HOG){

				CvRect cell;
				int x, y, cellw, cellh, cell_idx;
				int step = _hogsum[0].width;//这个step必须是hogsum的width

				hidfeature->misc.hog.inIdx = feature->misc.hog.outIdx % CV_HOG_BIN;

				x = feature->misc.hog.blocks.x;
				y = feature->misc.hog.blocks.y;
				cellw = feature->misc.hog.blocks.width / 2;
				cellh = feature->misc.hog.blocks.height / 2;

				cell_idx = feature->misc.hog.outIdx / CV_HOG_BIN;

				switch(cell_idx)
				{
				case 0:
					cell = cvRect(x, y, cellw, cellh);
					break;
				case 1:
					cell = cvRect(x+cellw, y, cellw, cellh);
					break;
				case 2:
					cell = cvRect(x, y+cellh, cellw, cellh);
					break;
				case 3:
					cell = cvRect(x+cellw, y+cellh, cellw, cellh);
				default :
					break;
				}

				CV_SUM_OFFSETS( hidfeature->misc.hog.fastRect.p0,
					hidfeature->misc.hog.fastRect.p1,
					hidfeature->misc.hog.fastRect.p2,
					hidfeature->misc.hog.fastRect.p3,
					cell, step )

			}
			else
				assert(0);

		} /* j */
	}
}


CV_INLINE
double icvCalcValue( CvHidMiscClassifier* classifier,size_t p_offset,size_t pq_offset,size_t hog_offset, CvMat* hogsum )
{
	double sum;
	double sqsum;
	int num = 0;
	float* hog_data;
	float ret = 0.0;
	CvHidMiscFeature hidfeature=classifier->features;
	switch(hidfeature.type)
	{
	case MISC_HAAR:
		{
			sum = calc_sum(hidfeature.misc.haar.rect[0],p_offset) * hidfeature.misc.haar.rect[0].weight;
			sum += calc_sum(hidfeature.misc.haar.rect[1],p_offset) * hidfeature.misc.haar.rect[1].weight;

			if( hidfeature.misc.haar.rect[2].weight != 0.0f )
				sum += calc_sum(hidfeature.misc.haar.rect[2],p_offset) * hidfeature.misc.haar.rect[2].weight;
			break;
		}
	case MISC_VAR:
		//variance feature doesn't need to change the threshold even if the size of feature is scaled.
		{
			int n = hidfeature.misc.var.rect[0].num_pixels;
			double avg,variance;
			assert(hidfeature.misc.var.used_rects ==1);
			sum = calc_sum(hidfeature.misc.var.rect[0],p_offset);
			avg = sum/n;
			sqsum = calc_sqsum(hidfeature.misc.var.rect[0],pq_offset);
			variance = (sqsum/n) - avg*avg;
			sum=variance;
		}
		break;
	case MISC_HOG:
		{
			num = hidfeature.misc.hog.inIdx;
			hog_data = (float*)hogsum[num].data.ptr;

			ret = hog_data[hidfeature.misc.hog.fastRect.p0 + hog_offset] 
			- hog_data[hidfeature.misc.hog.fastRect.p1 + hog_offset] 
			- hog_data[hidfeature.misc.hog.fastRect.p2 + hog_offset] 
			+ hog_data[hidfeature.misc.hog.fastRect.p3 + hog_offset];

			sum=ret;
		}
		break;			
	default:
		assert(0);
		break;

	}
	return sum;
}

double icvEvalHidMiscClassifier( int idx,double sum,CvHidMiscClassifier* classifier,
								double variance_norm_factor,
								CvRect windows,
								CvHidMiscClassifierCascade* cascade)
{
	double threshold,left,right;
	double result;

	CvHidMiscFeature feature = classifier->features;
	threshold=classifier->threshold[idx];
	left=classifier->left[idx];
	right=classifier->right[idx];
	double t1= threshold * variance_norm_factor;
	switch(feature.type)
	{
	case MISC_HAAR:
		result = sum < t1 ? left : right;
		break;
	case MISC_VAR:
		result = sum < threshold ? left :right;
		break;
	case MISC_HOG:
		result = sum < threshold ? left :right;
		break;			
	default:
		assert(0);
		break;

	}


	return result;
}                                 

vector<int> calced;
vector<double> calcvalues;
CV_IMPL int
cvRunMiscClassifierCascadeSum( const CvMiscClassifierCascade* _cascade,
							  CvPoint pt, CvMat* hogsum , double& stage_sum, int start_stage )
{
	int result = -1;

	int p_offset, pq_offset, hog_offset;
	int i, j;
	double mean, variance_norm_factor;
	CvHidMiscClassifierCascade* cascade;

	if( !CV_IS_MISC_CLASSIFIER(_cascade) )
		CV_Error( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid cascade pointer" );
	CvRect windows;

	int* group;
	int groupcount;
	int flag=0x7;
	int featureindex;
	int k;


	int stagesize;
	CvHidMiscStageClassifier *hidstageclassifier;
	CvHidMiscClassifier* hidclassifier;

	//pt(x,y) is detect window's coordinate
	windows.x=pt.x;
	windows.y=pt.y;
	windows.width=_cascade->orig_window_size.width;//50
	windows.height=_cascade->orig_window_size.height;//50
	cascade = _cascade->hid_cascade;
	if( !cascade )
		CV_Error( CV_StsNullPtr, "Hidden cascade has not been created.\n"
		"Use cvSetImagesForMiscClassifierCascade" );

	if( pt.x < 0 || pt.y < 0 ||
		pt.x + _cascade->real_window_size.width >= cascade->sum.width ||
		pt.y + _cascade->real_window_size.height >= cascade->sum.height )
		return -1;

	p_offset = pt.y * (cascade->sum.step/sizeof(sumtype)) + pt.x;//window's offset in img
	pq_offset = pt.y * (cascade->sqsum.step/sizeof(sqsumtype)) + pt.x;
	hog_offset = pt.y * (cascade->hogsum[0].step/sizeof(hogsumtype)) + pt.x;

	mean = calc_sum(*cascade,p_offset);
	variance_norm_factor = cascade->pq0[pq_offset] - cascade->pq1[pq_offset] -
		cascade->pq2[pq_offset] + cascade->pq3[pq_offset];
	variance_norm_factor = variance_norm_factor*cascade->inv_window_area - mean*mean;//检测框区域标准差

	if( variance_norm_factor >= 0. )
		variance_norm_factor = sqrt(variance_norm_factor);//和训练中的计算方法一样
	else												
		variance_norm_factor = 1.;		

	for( i = start_stage; i < cascade->count; i++ )//遍历强分类器
	{
		hidstageclassifier=&cascade->stage_classifier[i];
		hidclassifier=hidstageclassifier->classifier;//特征类型
		stagesize=hidstageclassifier->count;//一级强分类器的个数
		calced.clear();
		calced.resize(stagesize);
		calcvalues.clear();
		calcvalues.resize(stagesize);
		for(k=0;k<stagesize;k++)
			calced[k]=0;
		for(k=0;k<MAX_GROUPS;k++){
			stage_sum = 0.0;
			if((flag&(1<<k))==0)
				continue;
			group=hidstageclassifier->group[k];
			groupcount=hidstageclassifier->index[k];//每级强分类器中，每一类所包含的弱分类器个数
			for( j = 0; j < groupcount; j++ )
			{
				featureindex=group[j];
				if(calced[featureindex]==0)
				{
					calcvalues[featureindex]=icvCalcValue(hidclassifier+featureindex, p_offset,pq_offset,hog_offset,hogsum );
					calced[featureindex]=1;
				}
				stage_sum += icvEvalHidMiscClassifier(k,calcvalues[featureindex],hidclassifier+featureindex,variance_norm_factor,windows,cascade);
			}

			if( stage_sum < hidstageclassifier->subthreshold[k] ){
				flag=flag^(1<<k);
				if(flag==0)
					return (-1*i);
			}
		}
	}

	return 1;
}

CV_IMPL int
cvRunMiscClassifierCascade( const CvMiscClassifierCascade* _cascade,
						   CvPoint pt, CvMat* hogsum, int start_stage )
{
	double stage_sum;
	return cvRunMiscClassifierCascadeSum(_cascade, pt, hogsum, stage_sum, start_stage);
}

namespace cv
{

	struct MiscDetectObjects_ScaleImage_Invoker
	{
		MiscDetectObjects_ScaleImage_Invoker( const CvMiscClassifierCascade* _cascade,
			int _stripSize, double _factor,
			const Mat& _sum1, const Mat& _sqsum1, CvMat* _hogsum1, Mat* _norm1,
			Mat* _mask1, Rect _equRect, ConcurrentRectVector& _vec, 
			std::vector<int>& _levels, std::vector<double>& _weights,
			bool _outputLevels  )
		{
			cascade = _cascade;
			stripSize = _stripSize;//img.h-50-1/0
			factor = _factor;//1*1.1*1.1....
			sum1 = _sum1;
			sqsum1 = _sqsum1;
			norm1 = _norm1;
			mask1 = _mask1;
			equRect = _equRect;
			vec = &_vec;
			hogsum = _hogsum1;
		}

		void operator()( const BlockedRange& range ) const
		{
			Size winSize0 = cascade->orig_window_size;//50*50
			Size winSize(cvRound(winSize0.width*factor), cvRound(winSize0.height*factor));
			int y1 = range.begin()*stripSize, y2 = min(range.end()*stripSize, sum1.rows - 1 - winSize0.height);
			//y1=0,y2=480-50-1 .etc
			if (y2 <= y1 || sum1.cols <= 1 + winSize0.width)
				return;

			Size ssz(sum1.cols - 1 - winSize0.width, y2 - y1);//640-1-50,480-1-50
			int x, y, ystep = factor > 2 ? 1 : 2;

            //int coreNum = omp_get_num_procs();
		    //vector<vector<Rect>> tvec;
		    //tvec.resize(coreNum);
			//omp_set_num_threads(4);

			for( y = y1; y < y2; y += ystep ){
				//#pragma omp parallel for
				for( x = 0; x < ssz.width; x += ystep )//(x,y)是检测窗口的左上角顶点位置
				{
					double gypWeight;
					//int k = omp_get_thread_num();
					int result = cvRunMiscClassifierCascadeSum( cascade, cvPoint(x,y), hogsum, gypWeight, 0);
					{
						if( result > 0 )
							 //tvec[k].push_back(Rect(cvRound(x*factor), cvRound(y*factor),\
							                 winSize.width, winSize.height)); //每个线程往自己的数组里面写东西，不会引起并行冲突
							vec->push_back(Rect(cvRound(x*factor), cvRound(y*factor),winSize.width, winSize.height));
					}
				}
			}
			//for(int i=0;i<coreNum;i++){
				//vec->insert(vec->end(),tvec[i].begin(),tvec[i].end());
			//}

		}

		const CvMiscClassifierCascade* cascade;
		int stripSize;
		double factor;
		Mat sum1, sqsum1, *norm1, *mask1;
		CvMat *hogsum;
		Rect equRect;
		ConcurrentRectVector* vec;
	};


	struct MiscDetectObjects_ScaleCascade_Invoker
	{
		MiscDetectObjects_ScaleCascade_Invoker( const CvMiscClassifierCascade* _cascade,
			Size _winsize, const Range& _xrange, double _ystep,
			size_t _sumstep, const int** _p, const int** _pq,
			ConcurrentRectVector& _vec, CvMat* _hogsum )
		{
			cascade = _cascade;
			winsize = _winsize;
			xrange = _xrange;
			ystep = _ystep;
			sumstep = _sumstep;
			p = _p; pq = _pq;
			vec = &_vec;
			hogsum = _hogsum;
		}

		void operator()( const BlockedRange& range ) const
		{
			int iy, startY = range.begin(), endY = range.end();
			const int *p0 = p[0], *p1 = p[1], *p2 = p[2], *p3 = p[3];
			const int *pq0 = pq[0], *pq1 = pq[1], *pq2 = pq[2], *pq3 = pq[3];
			bool doCannyPruning = p0 != 0;
			int sstep = (int)(sumstep/sizeof(p0[0]));

			for( iy = startY; iy < endY; iy++ )
			{
				int ix, y = cvRound(iy*ystep), ixstep = 1;
				for( ix = xrange.start; ix < xrange.end; ix += ixstep )
				{
					int x = cvRound(ix*ystep); // it should really be ystep, not ixstep

					if( doCannyPruning )
					{
						int offset = y*sstep + x;
						int s = p0[offset] - p1[offset] - p2[offset] + p3[offset];
						int sq = pq0[offset] - pq1[offset] - pq2[offset] + pq3[offset];
						if( s < 100 || sq < 20 )
						{
							ixstep = 2;
							continue;
						}
					}

					int result = cvRunMiscClassifierCascade( cascade, cvPoint(x, y), hogsum, 0 );
					if( result > 0 )
						vec->push_back(Rect(x, y, winsize.width, winsize.height));
					ixstep = result != 0 ? 1 : 2;
				}
			}
		}

		const CvMiscClassifierCascade* cascade;
		double ystep;
		size_t sumstep;
		Size winsize;
		Range xrange;
		const int** p;
		const int** pq;
		ConcurrentRectVector* vec;
		CvMat* hogsum;
	};


}

int stagessum=0;
int  windowssum=0;
CvSeq*
cvMiscDetectObjectsForROC( const CvArr* _img, 
						  CvMiscClassifierCascade* cascade, CvMemStorage* storage,
						  std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
						  double scaleFactor, int minNeighbors, int flags, 
						  CvSize minSize, CvSize maxSize, bool outputRejectLevels )
{
	const double GROUP_EPS = 0.2;
	CvMat stub, *img = (CvMat*)_img;
	stagessum=0;
	windowssum=0;
	CvMat *temp, *sum, *sqsum, *imgSmall;
	CvMat  sum1, sqsum1, imgSmall1;

	CvMat* hogsum[CV_HOG_BIN];
	CvMat hogsum1[CV_HOG_BIN];

	CvSeq* result_seq = 0;
	cv::Ptr<CvMemStorage> temp_storage;
	CvRect equRect;
	cv::ConcurrentRectVector allCandidates;
	std::vector<cv::Rect> rectList;
	std::vector<int> rweights;
	double factor;
	int coi;

	if( !storage )
		CV_Error( CV_StsNullPtr, "Null storage pointer" );

	img = cvGetMat( img, &stub, &coi );
	if( coi )
		CV_Error( CV_BadCOI, "COI is not supported" );

	if( CV_MAT_DEPTH(img->type) != CV_8U )
		CV_Error( CV_StsUnsupportedFormat, "Only 8-bit images are supported" );

	if( scaleFactor <= 1 )
		CV_Error( CV_StsOutOfRange, "scale factor must be > 1" );


	if( maxSize.height == 0 || maxSize.width == 0 )//max_size is img's size
	{
		maxSize.height = img->rows;
		maxSize.width = img->cols;
	}

	temp = cvCreateMat( img->rows, img->cols, CV_8UC1 );
	if( !cascade->hid_cascade )
		icvCreateHidMiscClassifierCascade(cascade);

	result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );

	if( CV_MAT_CN(img->type) > 1 )
	{
		cvCvtColor( img, temp, CV_BGR2GRAY );
		img = temp;
	}

	cvReleaseMat(&temp);

	imgSmall = cvCreateMat( img->height , img->width , CV_8UC1 );
	sum = cvCreateMat( img->height + 1, img->width + 1, CV_32SC1 );
	sqsum = cvCreateMat( img->height + 1, img->width + 1, CV_64FC1 );
	for(int i = 0; i < CV_HOG_BIN; i++)
		hogsum[i] = cvCreateMat(img->height + 1, img->width + 1, CV_32FC1);


	if( flags & CV_HAAR_SCALE_IMAGE )
	{
		CvSize winSize0 = cascade->orig_window_size;//50*50

		for( factor = 1; ; factor *= scaleFactor )//scaleFactor = 1.1
		{
			CvSize sz = { cvRound( img->cols/factor ), cvRound( img->rows/factor ) };
			// sz resized from img(for example 640*480), this is real calculated img
			CvSize sz1 = { sz.width - winSize0.width + 1, sz.height - winSize0.height + 1 };

			initialize_mapbuf(sz.width, sz.height);

			imgSmall1 = cvMat( sz.height, sz.width, CV_8UC1, imgSmall->data.ptr );
			sum1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, sum->data.ptr );
			sqsum1 = cvMat( sz.height+1, sz.width+1, CV_64FC1, sqsum->data.ptr );

			for(int i = 0; i < CV_HOG_BIN; i++)
				hogsum1[i] = cvMat(sz.height+1, sz.width+1, CV_32FC1, hogsum[i]->data.ptr);


			if( sz1.width <= 0 || sz1.height <= 0 )
				break;


			cvResize( img, &imgSmall1, CV_INTER_LINEAR );
			cvIntegral( &imgSmall1, &sum1, &sqsum1);

			tHOGIntegral(&imgSmall1,hogsum1,CV_HOG_BIN);   //HOG

			int ystep = factor > 2 ? 1 : 2;
			const int stripCount = 1;
			//根据特征信息填充hid特征
			cvSetImagesForMiscClassifierCascade( cascade, &sum1, &sqsum1, hogsum1, 1. );            

			cv::Mat _norm1, _mask1;
			//cv::parallel_for(cv::BlockedRange(0, stripCount),\
				cv::MiscDetectObjects_ScaleImage_Invoker(cascade,\
				(((sz1.height + stripCount - 1)/stripCount + ystep-1)/ystep)*ystep,\
				factor, cv::Mat(&sum1), cv::Mat(&sqsum1), hogsum1, &_norm1, &_mask1,\
				cv::Rect(equRect), allCandidates, rejectLevels, levelWeights, outputRejectLevels));

			for( int y = 0; y < sz1.height; y += ystep )
			{
				for( int x = 0; x < sz1.width; x += ystep )
				{
					double gypWeight=0.;	
					windowssum++;
					int result = cvRunMiscClassifierCascadeSum( cascade, cvPoint(x,y), hogsum1,gypWeight, 0 );
					//int result=0;
					if( result > 0 )
					{
						allCandidates.push_back(Rect(cvRound(x*factor), cvRound(y*factor),cvRound(winSize0.width*factor),cvRound(winSize0.height*factor)));
					}
					else
					{
						stagessum+=result;
					}
				}
			}

		}
	}
	//printf("stagessum=%d,windowssum=%d,average=%f\n",stagessum,windowssum,(stagessum*1.0)/windowssum);
	fflush(stdout);

	cvReleaseMat(&imgSmall);
	cvReleaseMat(&sum);
	cvReleaseMat(&sqsum);
	for(int i=0;i<9;i++)
		cvReleaseMat(&hogsum[i]);


	rectList.resize(allCandidates.size());
	if(!allCandidates.empty())
		std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

	if( minNeighbors != 0 )
	{
		if( outputRejectLevels )
		{
			groupRectangles(rectList, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
		}
		else
		{
			groupRectangles(rectList, rweights, std::max(minNeighbors, 1), GROUP_EPS);
		}
	}
	else
		rweights.resize(rectList.size(),0);


	{
		for( size_t i = 0; i < rectList.size(); i++ )
		{
			CvAvgComp c;
			c.rect = rectList[i];
			c.neighbors = !rweights.empty() ? rweights[i] : 0;
			cvSeqPush( result_seq, &c );
		}
	}


	return result_seq;
}

CV_IMPL CvSeq*
cvMiscDetectObjects( const CvArr* _img, 
					CvMiscClassifierCascade* cascade, CvMemStorage* storage,
					double scaleFactor,
					int minNeighbors, int flags, CvSize minSize, CvSize maxSize )
{
	std::vector<int> fakeLevels;
	std::vector<double> fakeWeights;
	return cvMiscDetectObjectsForROC( _img, cascade, storage, fakeLevels, fakeWeights, 
		scaleFactor, minNeighbors, flags, minSize, maxSize, false );

}


static CvMiscClassifierCascade*
icvLoadCascadeCART( const char** input_cascade, int n, CvSize orig_window_size )
{
	int i;
	CvMiscClassifierCascade* cascade = icvCreateMiscClassifierCascade(n);
	cascade->orig_window_size = orig_window_size;

	for( i = 0; i < n; i++ )
	{
		int j, count, l;
		float threshold = 0;
		const char* stage = input_cascade[i];
		int dl = 0;

		/* tree links */
		int parent = -1;
		int next = -1;

		sscanf( stage, "%d%n", &count, &dl );
		stage += dl;

		assert( count > 0 );
		cascade->stage_classifier[i].count = count;
		cascade->stage_classifier[i].classifier =
			(CvMiscClassifier*)cvAlloc( count*sizeof(cascade->stage_classifier[i].classifier[0]));

		for( j = 0; j < count; j++ )
		{
			CvMiscClassifier* classifier = cascade->stage_classifier[i].classifier + j;
			int k, rects = 0;
			char str[100];

			sscanf( stage, "%d%n", &classifier->groupflag, &dl );
			stage += dl;

			int feature_type;
			sscanf( stage, "%s%d%n",str,&feature_type,&dl);
			stage += dl;


			switch(feature_type)
			{
			case MISC_HAAR:
				classifier->misc_feature.type = MISC_HAAR;
				sscanf( stage, "%d%n", &rects, &dl );
				stage += dl;

				assert( rects >= 2 && rects <= CV_HAAR_FEATURE_MAX );

				for( k = 0; k < rects; k++ )
				{
					CvRect r;
					int band = 0;
					sscanf( stage, "%d%d%d%d%d%f%n",
						&r.x, &r.y, &r.width, &r.height, &band,
						&(classifier->misc_feature.misc.haar.rect[k].weight), &dl );
					stage += dl;
					classifier->misc_feature.misc.haar.rect[k].r = r;
				}
				sscanf( stage, "%s%n", str, &dl );
				stage += dl;

				for( k = rects; k < CV_HAAR_FEATURE_MAX; k++ )
				{
					memset( classifier->misc_feature.misc.haar.rect + k, 0,
						sizeof(classifier->misc_feature.misc.haar.rect[k]) );
				}
				break;

			case MISC_VAR:
				classifier->misc_feature.type = MISC_VAR;
				sscanf( stage, "%d%n", &rects, &dl );
				stage += dl;

				assert( rects ==1 );//now only allow the number of rects being 1
				{
					CvRect r;
					int band = 0;
					sscanf( stage, "%d%d%d%d%n",
						&r.x, &r.y, &r.width, &r.height, &dl );
					stage += dl;
					classifier->misc_feature.misc.var.r0 = r;
				}
				classifier->misc_feature.misc.var.used_rects = rects;
				sscanf( stage, "%s%n", str, &dl );
				stage += dl;
				break;

			case  MISC_HOG:
				classifier->misc_feature.type = MISC_HOG;
				sscanf(stage,"%d%n",&classifier->misc_feature.misc.hog.outIdx,&dl);
				stage += dl;
				sscanf(stage,"%d%d%d%d%n",&classifier->misc_feature.misc.hog.blocks.x,
					&classifier->misc_feature.misc.hog.blocks.y,
					&classifier->misc_feature.misc.hog.blocks.width,
					&classifier->misc_feature.misc.hog.blocks.height,
					&dl);	                          
				stage+=dl;
				sscanf( stage, "%s%n", str, &dl );
				stage += dl;
				break;                          

			default:
				assert(0);
			}
			for( l = 0; l <MAX_GROUPS; l++ )
			{
				sscanf( stage, "%f%f%f%n", &(classifier->threshold[l]),
					&(classifier->left[l]),
					&(classifier->right[l]), &dl );
				stage += dl;
			}
		}



		for( j = 0; j < MAX_GROUPS; j++ )
		{
			int k;
			sscanf( stage, "%d%n", &count, &dl );
			stage += dl;
			cascade->stage_classifier[i].group[j].resize(count);
			for(k=0;k<count;k++)
			{
				sscanf( stage, "%d%n", &cascade->stage_classifier[i].group[j][k], &dl );
				stage += dl;				
			}
			sscanf( stage, "%f%n", &cascade->stage_classifier[i].subthreshold[j], &dl );
			stage += dl;	
		}

		/* load tree links */
		if( sscanf( stage, "%d%d%n", &parent, &next, &dl ) != 2 )
		{
			parent = i - 1;
			next = -1;
		}
		stage += dl;

		cascade->stage_classifier[i].parent = parent;
		cascade->stage_classifier[i].next = next;
		cascade->stage_classifier[i].child = -1;

		if( parent != -1 && cascade->stage_classifier[parent].child == -1 )
		{
			cascade->stage_classifier[parent].child = i;
		}
	}

	return cascade;
}

#ifndef _MAX_PATH
#define _MAX_PATH 1024
#endif

CV_IMPL CvMiscClassifierCascade*
cvLoadMiscClassifierCascade( const char* directory, CvSize orig_window_size )
{
	const char** input_cascade = 0;
	CvMiscClassifierCascade *cascade = 0;

	int i, n;
	const char* slash;
	char name[_MAX_PATH];
	int size = 0;
	char* ptr = 0;

	if( !directory )
		CV_Error( CV_StsNullPtr, "Null path is passed" );

	n = (int)strlen(directory)-1;
	slash = directory[n] == '\\' || directory[n] == '/' ? "" : "/";

	/* try to read the classifier from directory */
	for( n = 0; ; n++ )
	{
		sprintf( name, "%s%s%d/AdaBoostCARTMiscClassifier.txt", directory, slash, n );
		FILE* f = fopen( name, "rb" );
		if( !f )
			break;
		fseek( f, 0, SEEK_END );
		size += ftell( f ) + 1;
		fclose(f);
	}

	if( n == 0 && slash[0] )
		return (CvMiscClassifierCascade*)cvLoad( directory );

	if( n == 0 )
		CV_Error( CV_StsBadArg, "Invalid path" );

	size += (n+1)*sizeof(char*);
	input_cascade = (const char**)cvAlloc( size );
	ptr = (char*)(input_cascade + n + 1);

	for( i = 0; i < n; i++ )
	{
		sprintf( name, "%s/%d/AdaBoostCARTMiscClassifier.txt", directory, i );
		FILE* f = fopen( name, "rb" );
		if( !f )
			CV_Error( CV_StsError, "" );
		fseek( f, 0, SEEK_END );
		size = ftell( f );
		fseek( f, 0, SEEK_SET );
		fread( ptr, 1, size, f );
		fclose(f);
		input_cascade[i] = ptr;
		ptr += size;
		*ptr++ = '\0';
	}

	input_cascade[n] = 0;
	cascade = icvLoadCascadeCART( input_cascade, n, orig_window_size );

	if( input_cascade )
		cvFree( &input_cascade );

	return cascade;
}


CV_IMPL void
cvReleaseMiscClassifierCascade( CvMiscClassifierCascade** _cascade )
{
	if( _cascade && *_cascade )
	{
		int i, j;
		CvMiscClassifierCascade* cascade = *_cascade;

		for( i = 0; i < cascade->count; i++ )
		{
			cvFree( &cascade->stage_classifier[i].classifier );
		}
		icvReleaseHidMiscClassifierCascade( &cascade->hid_cascade );
		cvFree( _cascade );
	}
}


/****************************************************************************************\
*                                  Persistence functions                                 *
\****************************************************************************************/

/* field names */

#define ICV_HAAR_SIZE_NAME            "size"
#define ICV_HAAR_STAGES_NAME          "stages"
#define ICV_HAAR_TREES_NAME             "trees"

#define ICV_FEATURE_NAME              "feature"
#define ICV_FEATURE_TYPE_NAME         "misc_type"

#define ICV_HAAR_RECTS_NAME                 "haar_rects"
#define ICV_VAR_RECTS_NAME                  "var_rects"
#define ICV_HOG_RECTS_NAME                  "hog_rects"

#define ICV_HAAR_TILTED_NAME                "tilted"
#define ICV_HAAR_THRESHOLD_NAME           "threshold"
#define ICV_HAAR_LEFT_NODE_NAME           "left_node"
#define ICV_HAAR_LEFT_VAL_NAME            "left_val"
#define ICV_HAAR_RIGHT_NODE_NAME          "right_node"
#define ICV_HAAR_RIGHT_VAL_NAME           "right_val"
#define ICV_HAAR_STAGE_THRESHOLD_NAME   "stage_threshold"
#define ICV_HAAR_PARENT_NAME            "parent"
#define ICV_HAAR_NEXT_NAME              "next"

static int
icvIsMiscClassifier( const void* struct_ptr )
{
	return CV_IS_MISC_CLASSIFIER( struct_ptr );
}


void cvGetRectElements(CvFileNode* rects_fn,CvMiscFeature* misc_feature,int rect_flag,CvSize orig_window_size)
{
	//int l;
	//CvSeqReader rects_reader;
	//char buf[256];
	//float weight;
	//int compIdx=0;
	//if(rect_flag == MISC_HAAR){
	//	for( l=0;l<CV_HAAR_FEATURE_MAX;l++){
	//		misc_feature->misc.haar.rect[l].r=cvRect( 0, 0, 0, 0 );
	//		misc_feature->misc.haar.rect[l].weight=0;
	//	}
	//}
	//cvStartReadSeq(rects_fn->data.seq, &rects_reader);
	//for(l =0; l< rects_fn->data.seq->total; ++l)
	//{
	//	CvFileNode* rect_fn;
	//	CvFileNode* fn;
	//	CvRect r;
	//	rect_fn = (CvFileNode*) rects_reader.ptr;
	//	if(! CV_NODE_IS_SEQ(rect_fn->tag) || rect_fn->data.seq->total < 4)
	//	{
	//		sprintf( buf, "Rect %d is not a valid sequence.",l);
	//         	CV_Error( CV_StsError, buf );
	//	}
	//	fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 0 ); //x coordinate
	//	
	//  		if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i < 0 )
	//  		{
	//  			sprintf( buf, "x coordinate must be non-negative integer. ");
	//  			CV_Error( CV_StsError, buf );
	//  		}
	//  		r.x = fn->data.i;
	//  		
	//  		fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 1 ); //y coordinate
	//       if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i < 0 )
	//       {
	//           sprintf( buf, "y coordinate must be non-negative integer. ");
	//           CV_Error( CV_StsError, buf );
	//       }
	//       r.y = fn->data.i;
	//       
	//       fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 2 ); //width
	//       if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0
	//           || r.x + fn->data.i > orig_window_size.width )
	//       {
	//           sprintf( buf, "width must be positive integer and "
	//                    "(x + width) must not exceed window width. ");
	//           CV_Error( CV_StsError, buf );
	//       }
	//       r.width = fn->data.i;
	//       
	//       fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 3 ); //height
	//       if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0
	//           || r.y + fn->data.i > orig_window_size.height )
	//       {
	//           sprintf( buf, "height must be positive integer and "
	//                    "(y + height) must not exceed window height. ");
	//           CV_Error( CV_StsError, buf );
	//       }
	//       r.height = fn->data.i;
	//       
	//       if(rect_flag == MISC_HAAR){
	//       	fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 4 ); //weight
	//       	if( !CV_NODE_IS_REAL( fn->tag ) )
	//       	{
	//           	sprintf( buf, "weight must be real number. ");
	//           	CV_Error( CV_StsError, buf );
	//       	}
	//       	weight = (float) fn->data.f;
	//       }
	//       if(rect_flag == MISC_HOG){
	//       	fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 4 ); //weight
	//       	compIdx = fn->data.i;
	//       }
	//	if(rect_flag == MISC_HAAR){
	//       	misc_feature->misc.haar.rect[l].weight = weight;
	//       	misc_feature->misc.haar.rect[l].r = r;
	//       }
	//       else if(rect_flag == MISC_VAR){//MISC_VAR 
	//       	if( l ==0){
	//       		misc_feature->misc.var.r0 = r;
	//       		misc_feature->misc.var.used_rects = 1;
	//       	}
	//       	else{
	//       		misc_feature->misc.var.r1 = r;
	//       		misc_feature->misc.var.used_rects = 2;
	//       	}		
	//	}
	//	else{//MISC_HOG
	//		misc_feature->misc.hog.blocks=r;
	//		misc_feature->misc.hog.outIdx=compIdx;
	//	}

	//       CV_NEXT_SEQ_ELEM( sizeof( *rect_fn ), rects_reader );		
	// 	}
	return; 		

}

static void*
icvReadMiscClassifier( CvFileStorage* fs, CvFileNode* node )
{
	CvMiscClassifierCascade* cascade = NULL;

	//char buf[256];
	//CvFileNode* seq_fn = NULL; /* sequence */
	//CvFileNode* fn = NULL;
	//CvFileNode* stages_fn = NULL;
	//CvSeqReader stages_reader;
	//int n;
	//int i, j, k;
	//int parent, next;

	//stages_fn = cvGetFileNodeByName( fs, node, ICV_HAAR_STAGES_NAME );
	//if( !stages_fn || !CV_NODE_IS_SEQ( stages_fn->tag) )
	//    CV_Error( CV_StsError, "Invalid stages node" );

	//n = stages_fn->data.seq->total;
	//cascade = icvCreateMiscClassifierCascade(n);

	///* read size */
	//seq_fn = cvGetFileNodeByName( fs, node, ICV_HAAR_SIZE_NAME );
	//if( !seq_fn || !CV_NODE_IS_SEQ( seq_fn->tag ) || seq_fn->data.seq->total != 2 )
	//    CV_Error( CV_StsError, "size node is not a valid sequence." );
	//fn = (CvFileNode*) cvGetSeqElem( seq_fn->data.seq, 0 );
	//if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0 )
	//    CV_Error( CV_StsError, "Invalid size node: width must be positive integer" );
	//cascade->orig_window_size.width = fn->data.i;
	//fn = (CvFileNode*) cvGetSeqElem( seq_fn->data.seq, 1 );
	//if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0 )
	//    CV_Error( CV_StsError, "Invalid size node: height must be positive integer" );
	//cascade->orig_window_size.height = fn->data.i;

	//cvStartReadSeq( stages_fn->data.seq, &stages_reader );
	//for( i = 0; i < n; ++i )
	//{
	//    CvFileNode* stage_fn;
	//    CvFileNode* trees_fn;
	//    CvSeqReader trees_reader;

	//    stage_fn = (CvFileNode*) stages_reader.ptr;
	//    if( !CV_NODE_IS_MAP( stage_fn->tag ) )
	//    {
	//        sprintf( buf, "Invalid stage %d", i );
	//        CV_Error( CV_StsError, buf );
	//    }

	//    trees_fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_TREES_NAME );
	//    if( !trees_fn || !CV_NODE_IS_SEQ( trees_fn->tag )
	//        || trees_fn->data.seq->total <= 0 )
	//    {
	//        sprintf( buf, "Trees node is not a valid sequence. (stage %d)", i );
	//        CV_Error( CV_StsError, buf );
	//    }

	//    cascade->stage_classifier[i].classifier =
	//        (CvMiscClassifier*) cvAlloc( trees_fn->data.seq->total
	//            * sizeof( cascade->stage_classifier[i].classifier[0] ) );
	//    for( j = 0; j < trees_fn->data.seq->total; ++j )
	//    {
	//        cascade->stage_classifier[i].classifier[j].misc_feature = NULL;
	//    }
	//    cascade->stage_classifier[i].count = trees_fn->data.seq->total;

	//    cvStartReadSeq( trees_fn->data.seq, &trees_reader );
	//    for( j = 0; j < trees_fn->data.seq->total; ++j )
	//    {
	//        CvFileNode* tree_fn;
	//        CvSeqReader tree_reader;
	//        CvMiscClassifier* classifier;
	//        int last_idx;

	//        classifier = &cascade->stage_classifier[i].classifier[j];
	//        tree_fn = (CvFileNode*) trees_reader.ptr;
	//        if( !CV_NODE_IS_SEQ( tree_fn->tag ) || tree_fn->data.seq->total <= 0 )
	//        {
	//            sprintf( buf, "Tree node is not a valid sequence."
	//                     " (stage %d, tree %d)", i, j );
	//            CV_Error( CV_StsError, buf );
	//        }

	//        classifier->count = tree_fn->data.seq->total;
	//        classifier->misc_feature = (CvMiscFeature*) cvAlloc(
	//            classifier->count * ( sizeof( *classifier->misc_feature ) +
	//                                  sizeof( *classifier->threshold ) +
	//                                  sizeof( *classifier->left ) +
	//                                  sizeof( *classifier->right ) ) +
	//            (classifier->count + 1) * sizeof( *classifier->alpha ) );
	//        classifier->threshold = (float*) (classifier->misc_feature+classifier->count);
	//        classifier->left = (int*) (classifier->threshold + classifier->count);
	//        classifier->right = (int*) (classifier->left + classifier->count);
	//        classifier->alpha = (float*) (classifier->right + classifier->count);

	//        cvStartReadSeq( tree_fn->data.seq, &tree_reader );
	//        for( k = 0, last_idx = 0; k < tree_fn->data.seq->total; ++k )
	//        {
	//            CvFileNode* node_fn;
	//            CvFileNode* feature_fn;
	//            CvFileNode* rects_fn;
	//            CvFileNode* type_fn;

	//            node_fn = (CvFileNode*) tree_reader.ptr;
	//            if( !CV_NODE_IS_MAP( node_fn->tag ) )
	//            {
	//                sprintf( buf, "Tree node %d is not a valid map. (stage %d, tree %d)",
	//                         k, i, j );
	//                CV_Error( CV_StsError, buf );
	//            }
	//            feature_fn = cvGetFileNodeByName( fs, node_fn, ICV_FEATURE_NAME );
	//            if( !feature_fn || !CV_NODE_IS_MAP( feature_fn->tag ) )
	//            {
	//                sprintf( buf, "Feature node is not a valid map. "
	//                         "(stage %d, tree %d, node %d)", i, j, k );
	//                CV_Error( CV_StsError, buf );
	//            }
	//            
	//            type_fn = cvGetFileNodeByName( fs,feature_fn,ICV_FEATURE_TYPE_NAME);
	//            if(!type_fn || !CV_NODE_IS_INT( type_fn->tag ) )
	//            {
	//            	sprintf( buf, "feature type must be 1(MISC_HAAR) or 2(MISC_VAR). "
	//                         "(stage %d, tree %d, node %d)", i, j, k );
	//                CV_Error( CV_StsError, buf );
	//            }
	//            switch(type_fn->data.i)
	//            {
	//	case MISC_HAAR:
	//		
	//		classifier->misc_feature[k].type = MISC_HAAR;
	//		rects_fn = cvGetFileNodeByName(fs,feature_fn,ICV_HAAR_RECTS_NAME);
	//	
	//		if( !rects_fn || !CV_NODE_IS_SEQ( rects_fn->tag )
	//                      || rects_fn->data.seq->total < 1
	//                      || rects_fn->data.seq->total > CV_HAAR_FEATURE_MAX )
	//                    {
	//                		sprintf( buf, "Rects node is not a valid haar rect sequence. "
	//                         "(stage %d, tree %d, node %d)", i, j, k );
	//                		CV_Error( CV_StsError, buf );
	//                    }
	//		
	//		cvGetRectElements(rects_fn,&classifier->misc_feature[k],MISC_HAAR,cascade->orig_window_size);
	//		
	//		break;
	//		
	//	case MISC_VAR:
	//		classifier->misc_feature[k].type = MISC_VAR;
	//		rects_fn = cvGetFileNodeByName(fs,feature_fn,ICV_VAR_RECTS_NAME);
	//	
	//		if( !rects_fn || !CV_NODE_IS_SEQ( rects_fn->tag )
	//                      || rects_fn->data.seq->total < 1
	//                      || rects_fn->data.seq->total > 2 )
	//                    {
	//                		sprintf( buf, "Rects node is not a valid var rect sequence. "
	//                         "(stage %d, tree %d, node %d)", i, j, k );
	//                		CV_Error( CV_StsError, buf );
	//                    }
	//		
	//		cvGetRectElements(rects_fn,&classifier->misc_feature[k],MISC_VAR,cascade->orig_window_size);
	//		break;
	//	case MISC_HOG:
	//		classifier->misc_feature[k].type = MISC_HOG;
	//		rects_fn = cvGetFileNodeByName(fs,feature_fn,ICV_HOG_RECTS_NAME);
	//	
	//		if( !rects_fn || !CV_NODE_IS_SEQ( rects_fn->tag )
	//                      || rects_fn->data.seq->total < 1
	//                      || rects_fn->data.seq->total > 2 )
	//                    {
	//                		sprintf( buf, "Rects node is not a valid hog rect sequence. "
	//                         "(stage %d, tree %d, node %d)", i, j, k );
	//                		CV_Error( CV_StsError, buf );
	//                    }
	//		
	//		cvGetRectElements(rects_fn,&classifier->misc_feature[k],MISC_HOG,cascade->orig_window_size);						
	//		break;
	//		
	//	default:
	//		assert(0);                
	//            }
	//         
	//            fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_THRESHOLD_NAME);
	//            if( !fn || !CV_NODE_IS_REAL( fn->tag ) )
	//            {
	//                sprintf( buf, "threshold must be real number. "
	//                         "(stage %d, tree %d, node %d)", i, j, k );
	//                CV_Error( CV_StsError, buf );
	//            }
	//            classifier->threshold[k] = (float) fn->data.f;
	//            fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_LEFT_NODE_NAME);
	//            if( fn )
	//            {
	//                if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= k
	//                    || fn->data.i >= tree_fn->data.seq->total )
	//                {
	//                    sprintf( buf, "left node must be valid node number. "
	//                             "(stage %d, tree %d, node %d)", i, j, k );
	//                    CV_Error( CV_StsError, buf );
	//                }
	//                /* left node */
	//                classifier->left[k] = fn->data.i;
	//            }
	//            else
	//            {
	//                fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_LEFT_VAL_NAME );
	//                if( !fn )
	//                {
	//                    sprintf( buf, "left node or left value must be specified. "
	//                             "(stage %d, tree %d, node %d)", i, j, k );
	//                    CV_Error( CV_StsError, buf );
	//                }
	//                if( !CV_NODE_IS_REAL( fn->tag ) )
	//                {
	//                    sprintf( buf, "left value must be real number. "
	//                             "(stage %d, tree %d, node %d)", i, j, k );
	//                    CV_Error( CV_StsError, buf );
	//                }
	//                /* left value */
	//                if( last_idx >= classifier->count + 1 )
	//                {
	//                    sprintf( buf, "Tree structure is broken: too many values. "
	//                             "(stage %d, tree %d, node %d)", i, j, k );
	//                    CV_Error( CV_StsError, buf );
	//                }
	//                classifier->left[k] = -last_idx;
	//                classifier->alpha[last_idx++] = (float) fn->data.f;
	//            }
	//            fn = cvGetFileNodeByName( fs, node_fn,ICV_HAAR_RIGHT_NODE_NAME);
	//            if( fn )
	//            {
	//                if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= k
	//                    || fn->data.i >= tree_fn->data.seq->total )
	//                {
	//                    sprintf( buf, "right node must be valid node number. "
	//                             "(stage %d, tree %d, node %d)", i, j, k );
	//                    CV_Error( CV_StsError, buf );
	//                }
	//                /* right node */
	//                classifier->right[k] = fn->data.i;
	//            }
	//            else
	//            {
	//                fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_RIGHT_VAL_NAME );
	//                if( !fn )
	//                {
	//                    sprintf( buf, "right node or right value must be specified. "
	//                             "(stage %d, tree %d, node %d)", i, j, k );
	//                    CV_Error( CV_StsError, buf );
	//                }
	//                if( !CV_NODE_IS_REAL( fn->tag ) )
	//                {
	//                    sprintf( buf, "right value must be real number. "
	//                             "(stage %d, tree %d, node %d)", i, j, k );
	//                    CV_Error( CV_StsError, buf );
	//                }
	//                /* right value */
	//                if( last_idx >= classifier->count + 1 )
	//                {
	//                    sprintf( buf, "Tree structure is broken: too many values. "
	//                             "(stage %d, tree %d, node %d)", i, j, k );
	//                    CV_Error( CV_StsError, buf );
	//                }
	//                classifier->right[k] = -last_idx;
	//                classifier->alpha[last_idx++] = (float) fn->data.f;
	//            }

	//            CV_NEXT_SEQ_ELEM( sizeof( *node_fn ), tree_reader );
	//        } /* for each node */
	//        if( last_idx != classifier->count + 1 )
	//        {
	//            sprintf( buf, "Tree structure is broken: too few values. "
	//                     "(stage %d, tree %d)", i, j );
	//            CV_Error( CV_StsError, buf );
	//        }

	//        CV_NEXT_SEQ_ELEM( sizeof( *tree_fn ), trees_reader );
	//    } /* for each tree */

	//    fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_STAGE_THRESHOLD_NAME);
	//    if( !fn || !CV_NODE_IS_REAL( fn->tag ) )
	//    {
	//        sprintf( buf, "stage threshold must be real number. (stage %d)", i );
	//        CV_Error( CV_StsError, buf );
	//    }
	//    cascade->stage_classifier[i].threshold = (float) fn->data.f;

	//    parent = i - 1;
	//    next = -1;

	//    fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_PARENT_NAME );
	//    if( !fn || !CV_NODE_IS_INT( fn->tag )
	//        || fn->data.i < -1 || fn->data.i >= cascade->count )
	//    {
	//        sprintf( buf, "parent must be integer number. (stage %d)", i );
	//        CV_Error( CV_StsError, buf );
	//    }
	//    parent = fn->data.i;
	//    fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_NEXT_NAME );
	//    if( !fn || !CV_NODE_IS_INT( fn->tag )
	//        || fn->data.i < -1 || fn->data.i >= cascade->count )
	//    {
	//        sprintf( buf, "next must be integer number. (stage %d)", i );
	//        CV_Error( CV_StsError, buf );
	//    }
	//    next = fn->data.i;

	//    cascade->stage_classifier[i].parent = parent;
	//    cascade->stage_classifier[i].next = next;
	//    cascade->stage_classifier[i].child = -1;

	//    if( parent != -1 && cascade->stage_classifier[parent].child == -1 )
	//    {
	//        cascade->stage_classifier[parent].child = i;
	//    }

	//    CV_NEXT_SEQ_ELEM( sizeof( *stage_fn ), stages_reader );
	//} /* for each stage */

	return cascade;
}

static void
icvWriteMiscClassifier( CvFileStorage* fs, const char* name, const void* struct_ptr,
					   CvAttrList attributes )
{
	//int i, j, k, l;
	//char buf[256];
	//const CvMiscClassifierCascade* cascade = (const CvMiscClassifierCascade*) struct_ptr;

	///* TODO: parameters check */

	//cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_MISC, attributes );

	//cvStartWriteStruct( fs, ICV_HAAR_SIZE_NAME, CV_NODE_SEQ | CV_NODE_FLOW );
	//cvWriteInt( fs, NULL, cascade->orig_window_size.width );
	//cvWriteInt( fs, NULL, cascade->orig_window_size.height );
	//cvEndWriteStruct( fs ); /* size */

	//cvStartWriteStruct( fs, ICV_HAAR_STAGES_NAME, CV_NODE_SEQ );
	//for( i = 0; i < cascade->count; ++i )
	//{
	//    cvStartWriteStruct( fs, NULL, CV_NODE_MAP );
	//    sprintf( buf, "stage %d", i );
	//    cvWriteComment( fs, buf, 1 );

	//    cvStartWriteStruct( fs, ICV_HAAR_TREES_NAME, CV_NODE_SEQ );

	//    for( j = 0; j < cascade->stage_classifier[i].count; ++j )
	//    {
	//        CvMiscClassifier* tree = &cascade->stage_classifier[i].classifier[j];

	//        cvStartWriteStruct( fs, NULL, CV_NODE_SEQ );
	//        sprintf( buf, "tree %d", j );
	//        cvWriteComment( fs, buf, 1 );

	//        for( k = 0; k < tree->count; ++k )
	//        {
	//            CvMiscFeature* feature = &tree->misc_feature[k];

	//            cvStartWriteStruct( fs, NULL, CV_NODE_MAP );
	//            if( k )
	//            {
	//                sprintf( buf, "node %d", k );
	//            }
	//            else
	//            {
	//                sprintf( buf, "root node" );
	//            }
	//            cvWriteComment( fs, buf, 1 );

	//            cvStartWriteStruct( fs, ICV_FEATURE_NAME, CV_NODE_MAP );//feature
	//cvWriteInt( fs, ICV_FEATURE_TYPE_NAME, feature->type );//feature_type
	//switch(feature->type){
	//	case MISC_HAAR:
	//		cvStartWriteStruct( fs, ICV_HAAR_RECTS_NAME, CV_NODE_SEQ );
	//           			for( l = 0; l < CV_HAAR_FEATURE_MAX && feature->misc.haar.rect[l].r.width != 0; ++l )
	//            		{
	//                		cvStartWriteStruct( fs, NULL, CV_NODE_SEQ | CV_NODE_FLOW );
	//                		cvWriteInt(  fs, NULL, feature->misc.haar.rect[l].r.x );
	//                		cvWriteInt(  fs, NULL, feature->misc.haar.rect[l].r.y );
	//                		cvWriteInt(  fs, NULL, feature->misc.haar.rect[l].r.width );
	//                		cvWriteInt(  fs, NULL, feature->misc.haar.rect[l].r.height );
	//                		cvWriteReal( fs, NULL, feature->misc.haar.rect[l].weight );
	//                		cvEndWriteStruct( fs ); /* rect */
	//            		}
	//            		cvEndWriteStruct( fs ); /* rects */
	//		break;
	//		
	//	case MISC_VAR:
	//		cvStartWriteStruct( fs, ICV_VAR_RECTS_NAME,CV_NODE_SEQ);
	//		assert(feature->misc.var.used_rects == 1);//only support one rect of variance feature. Li
	//		
	//		{
	//			cvStartWriteStruct( fs, NULL, CV_NODE_SEQ | CV_NODE_FLOW );
	//                		cvWriteInt(  fs, NULL, feature->misc.var.r0.x );
	//                		cvWriteInt(  fs, NULL, feature->misc.var.r0.y );
	//                		cvWriteInt(  fs, NULL, feature->misc.var.r0.width );
	//                		cvWriteInt(  fs, NULL, feature->misc.var.r0.height );
	//                		cvEndWriteStruct( fs ); /* rect */
	//		}
	//		cvEndWriteStruct( fs ); /* rects */
	//		break;
	//	

	//	case MISC_HOG:
	//		cvStartWriteStruct( fs, ICV_HOG_RECTS_NAME,CV_NODE_SEQ);
	//		{
	//			cvStartWriteStruct( fs, NULL, CV_NODE_SEQ | CV_NODE_FLOW );
	//                		cvWriteInt(  fs, NULL, feature->misc.hog.blocks.x );
	//                		cvWriteInt(  fs, NULL, feature->misc.hog.blocks.y );
	//                		cvWriteInt(  fs, NULL, feature->misc.hog.blocks.width );
	//                		cvWriteInt(  fs, NULL, feature->misc.hog.blocks.height );
	//			cvWriteInt(  fs, NULL, feature->misc.hog.outIdx);
	//                		cvEndWriteStruct( fs ); /* rect */
	//		}
	//		cvEndWriteStruct( fs ); /* rects */	
	//		break;
	//	default:
	//		assert( 0 );
	//}
	//
	//            cvEndWriteStruct( fs ); /* misc feature */

	//            cvWriteReal( fs, ICV_HAAR_THRESHOLD_NAME, tree->threshold[k]);

	//            if( tree->left[k] > 0 )
	//            {
	//                cvWriteInt( fs, ICV_HAAR_LEFT_NODE_NAME, tree->left[k] );
	//            }
	//            else
	//            {
	//                cvWriteReal( fs, ICV_HAAR_LEFT_VAL_NAME,
	//                    tree->alpha[-tree->left[k]] );
	//            }

	//            if( tree->right[k] > 0 )
	//            {
	//                cvWriteInt( fs, ICV_HAAR_RIGHT_NODE_NAME, tree->right[k] );
	//            }
	//            else
	//            {
	//                cvWriteReal( fs, ICV_HAAR_RIGHT_VAL_NAME,
	//                    tree->alpha[-tree->right[k]] );
	//            }

	//            cvEndWriteStruct( fs ); /* split */
	//        }

	//        cvEndWriteStruct( fs ); /* tree */
	//    }

	//    cvEndWriteStruct( fs ); /* trees */

	//    cvWriteReal( fs, ICV_HAAR_STAGE_THRESHOLD_NAME, cascade->stage_classifier[i].threshold);
	//    cvWriteInt( fs, ICV_HAAR_PARENT_NAME, cascade->stage_classifier[i].parent );
	//    cvWriteInt( fs, ICV_HAAR_NEXT_NAME, cascade->stage_classifier[i].next );

	//    cvEndWriteStruct( fs ); /* stage */
	//} /* for each stage */

	//cvEndWriteStruct( fs ); /* stages */
	//cvEndWriteStruct( fs ); /* root */
}

static void*
icvCloneMiscClassifier( const void* struct_ptr )
{
	CvMiscClassifierCascade* cascade = NULL;

	/*  int i, j, k, n;
	const CvMiscClassifierCascade* cascade_src =
	(const CvMiscClassifierCascade*) struct_ptr;

	n = cascade_src->count;
	cascade = icvCreateMiscClassifierCascade(n);
	cascade->orig_window_size = cascade_src->orig_window_size;

	for( i = 0; i < n; ++i )
	{
	cascade->stage_classifier[i].parent = cascade_src->stage_classifier[i].parent;
	cascade->stage_classifier[i].next = cascade_src->stage_classifier[i].next;
	cascade->stage_classifier[i].child = cascade_src->stage_classifier[i].child;
	cascade->stage_classifier[i].threshold = cascade_src->stage_classifier[i].threshold;

	cascade->stage_classifier[i].count = 0;
	cascade->stage_classifier[i].classifier =
	(CvMiscClassifier*) cvAlloc( cascade_src->stage_classifier[i].count
	* sizeof( cascade->stage_classifier[i].classifier[0] ) );

	cascade->stage_classifier[i].count = cascade_src->stage_classifier[i].count;

	for( j = 0; j < cascade->stage_classifier[i].count; ++j )
	cascade->stage_classifier[i].classifier[j].misc_feature = NULL;

	for( j = 0; j < cascade->stage_classifier[i].count; ++j )
	{
	const CvMiscClassifier* classifier_src =
	&cascade_src->stage_classifier[i].classifier[j];
	CvMiscClassifier* classifier =
	&cascade->stage_classifier[i].classifier[j];

	classifier->count = classifier_src->count;
	classifier->misc_feature = (CvMiscFeature*) cvAlloc(
	classifier->count * ( sizeof( *classifier->misc_feature ) +
	sizeof( *classifier->threshold ) +
	sizeof( *classifier->left ) +
	sizeof( *classifier->right ) ) +
	(classifier->count + 1) * sizeof( *classifier->alpha ) );
	classifier->threshold = (float*) (classifier->misc_feature+classifier->count);
	classifier->left = (int*) (classifier->threshold + classifier->count);
	classifier->right = (int*) (classifier->left + classifier->count);
	classifier->alpha = (float*) (classifier->right + classifier->count);
	for( k = 0; k < classifier->count; ++k )
	{
	classifier->misc_feature[k] = classifier_src->misc_feature[k];
	classifier->threshold[k] = classifier_src->threshold[k];
	classifier->left[k] = classifier_src->left[k];
	classifier->right[k] = classifier_src->right[k];
	classifier->alpha[k] = classifier_src->alpha[k];
	}
	classifier->alpha[classifier->count] =
	classifier_src->alpha[classifier->count];
	}
	}*/

	return cascade;
}


CvType haar_type( CV_TYPE_NAME_MISC, icvIsMiscClassifier,
				 (CvReleaseFunc)cvReleaseMiscClassifierCascade,
				 icvReadMiscClassifier, icvWriteMiscClassifier,
				 icvCloneMiscClassifier );

/* End of file. */

