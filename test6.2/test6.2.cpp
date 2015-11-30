#include "stdafx.h"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

int main(int argc, char* argv[]) {

	CvHaarClassifierCascade *pCascadeFrontal = 0, *pCascadeProfile = 0;	
	CvMemStorage *pStorage = 0;	
	CvSeq *pFaceRectSeq;
	int i;

	IplImage *pInpImg = cvLoadImage("D:/≤‚ ‘/test6.2/6.jpg", CV_LOAD_IMAGE_COLOR);
	pStorage = cvCreateMemStorage(0);

	pCascadeFrontal = (CvHaarClassifierCascade *) cvLoad ("D:/≤‚ ‘/test6.2/haarcascade/haarcascade_frontalface_default.xml",0,0,0);
	pCascadeProfile = (CvHaarClassifierCascade *) cvLoad ("D:/≤‚ ‘/test6.2/haarcascade/haarcascade_profileface.xml",0,0,0);

	if (!pInpImg || !pStorage || !pCascadeFrontal || !pCascadeProfile) {
		printf("L'initilisation a echoue");
		exit(-1);
	}

	cvNamedWindow("Fenetre de Haar", CV_WINDOW_NORMAL);
	cvShowImage("Fenetre de Haar", pInpImg);
	cvWaitKey(50);

	pFaceRectSeq = cvHaarDetectObjects
		(pInpImg, pCascadeFrontal, pStorage,1.1,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(0, 0));	

	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0);
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}
	cvShowImage("Fenetre de Haar", pInpImg);
	cvWaitKey(1);
	//≤‡¡≥
	pFaceRectSeq = cvHaarDetectObjects
		(pInpImg, pCascadeProfile, pStorage,
		1.4,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(0, 0));

	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(255,165,0), 3, 4, 0);
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}

	cvShowImage("Fenetre de Haar", pInpImg);
	cvWaitKey(0);
	cvDestroyWindow("Fenetre de Haar");

	cvReleaseImage(&pInpImg);
	if (pCascadeFrontal) cvReleaseHaarClassifierCascade(&pCascadeFrontal);
	if (pCascadeProfile) cvReleaseHaarClassifierCascade(&pCascadeProfile);
	if (pStorage) cvReleaseMemStorage(&pStorage);
}