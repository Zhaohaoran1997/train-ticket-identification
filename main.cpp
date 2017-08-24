// author: Ying Jia
// time: 20170824
// contact: 1439352516@qq.com
// 功能：实现对自然场景中火车票的定位识别校正

#include <fstream>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <string>
#include "dingwei/text_hancer.h"
#include "dingwei/text_binarizer.h"
#include "shibie/stdafx.h"
#include "shibie/QRCodeReader.h"
#include "shibie/common/BitMatrix.h"
#include "shibie/qrcode/detector/Detector.h"
#include "shibie/Binarizer.h"
#include "shibie/LuminanceSource.h"
#include <vector>
#include "shibie/common/GreyscaleLuminanceSource.h"
#include "shibie/common/GlobalHistogramBinarizer.h"
#include "shibie/Exception.h"

using namespace cv;
using namespace std;
using namespace yingjia;
using namespace zxing;

/*-------------------批量载入图片-------------------------*/
void loadlist(string filename, vector<string>& list)
{
	ifstream infile(filename.c_str(), std::ios::in);
	string line;
	while(infile>>line)
	{
		list.push_back(line);
	}
	infile.close();
}

int main(int argc, char** argv)
{
	if(argc!=3)
	{
		std::cout<<"main.exe <inlist> <outdir>\n";
		return 1;
	}
	vector<string> list;
	loadlist(string(argv[1]),list);
	for(int i=0; i<list.size(); i++)
	{
		/*------------------------寻找整张图片中的二维码区域----------------------------*/
		Mat srcImg=imread(list[i], 1);
		Mat grayImg;
		cvtColor(srcImg, grayImg, CV_BGR2GRAY);
		Mat erzhiImg;
		yingjia::TextBinarizer erzhi;
		erzhi.Run2(grayImg, erzhiImg);
		Mat element=getStructuringElement(MORPH_ELLIPSE, Size(20,20),Point(-1,-1));
		Mat tp;
		dilate(erzhiImg, tp, element);
		vector<vector<Point2i> > contours;
		findContours(tp.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		double maxArea=0.0;
		RotatedRect rotarect(Point2f(0,0), Size2f(0,0),0);
		Point2f vertices[4];
		Mat srcTmp=erzhiImg.clone();
		int temp=0;
		for(int k=0; k<contours.size(); k++)
		{
			rotarect=minAreaRect(contours[k]);
			float flag=(float)rotarect.size.width / (float)rotarect.size.height;
			if((flag>0.9) && (flag<1.1))
			{
				double area=fabs(contourArea(contours[k]));
				if(area>maxArea)
				{
					temp=k;
					maxArea=area;
					rotarect.points(vertices);
				}
			}

		}
		Mat savedGrayMat=Mat::zeros(srcImg.rows, srcImg.cols, CV_8UC1);
		drawContours(savedGrayMat, contours, temp, Scalar(255),CV_FILLED);
		vector<vector<Point2i> > newcontours;
		findContours(savedGrayMat.clone(), newcontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		Rect rect(0,0,0,0);
		rect=boundingRect(newcontours[0]);
		Mat roi;
		roi=srcImg(Rect(rect.x, rect.y, rect.width, rect.height));
		/*---------------------------------------利用zxing源码识别二维码上三个定位点坐标----------------------*/
		Mat cv_img=roi;
		int rows=cv_img.rows;
		int cols=cv_img.cols;
		cvtColor(cv_img, cv_img, CV_BGR2GRAY);
		Array<char> source1(rows*cols);
		int count=0;
		for(int j=0; j<rows; j++)
		{
			for(int i=0; i<cols; i++)
			{
				source1[count++]=cv_img.at<uchar>(j,i);
			}
		}
		ArrayRef<char> source(&source1);
		LuminanceSource *qrGreyLuminance=new GreyscaleLuminanceSource(source, cols, rows, 0,0, cols,rows);
		Ref<LuminanceSource> lusource(qrGreyLuminance->rotateCounterClockwise());
		Binarizer *qrBinary=new GlobalHistogramBinarizer(lusource);
		Ref<BitMatrix> bit_inst;
		try {
			bit_inst.reset(qrBinary->getBlackMatrix());
		}
		catch (zxing::Exception& e) {
			std::cout<<e.what()<<std::endl;
			continue;
		}
		DecodeHints hints(DecodeHints::QR_CODE_HINT);
		zxing::qrcode::Detector detector_inst;
		detector_inst.setImage(bit_inst);
		Ref<DetectorResult> result;
		try {
			result.reset(detector_inst.detect(hinst));
		}
		catch (zxing::Exception& e) {
			std::cout<<e.what()<<std::endl;
			continue;
		}
		ArrayRef<Ref<ResultPoint> > points(result->getPoints());
		ResultPoint s(points[2][0]);
		vector<Point2f> res;
		res.push_back(Point2f(cols-points[0][0].getY(), points[0][0].getX()));
		res.push_back(Point2f(cols-points[1][0].getY(), points[1][0].getX()));
		res.push_back(Point2f(cols-points[2][0].getY(), points[2][0].getX()));
		/*------------------------利用仿射变换校正火车票---------------------------------*/
		Mat dstColorImg;
		int templateW=1018;
		int templateH=650;
		vector<Point2f> srcBlocks, dstBlocks;
		srcBlocks.push_back(res[0]);
		srcBlocks.push_back(res[1]);
		srcBlocks.push_back(res[2]);
		dstBlocks.push_back(Point2f(787,545));
		dstBlocks.push_back(Point2f(787,413));
		dstBlocks.push_back(Point2f(920,413));
    cv::Size dstSize(templateW, templateH);
    Mat warp_mat = getAffineTransform(&srcBlocks.front(), &dstBlocks.front());
    warpAffine(srcImg, dstColorImg, warp_mat, dstSize, INTER_CUBIC);
		char save_img_name[1024];
		_snprintf_s(save_img_name, 1024, "%s\\%08d.jpg", argv[2], i);
		imwrite(save_img_name, dstColorImg);
		std::cout<<save_img_name<<" processed.\n";
	}
	return 0;
}
