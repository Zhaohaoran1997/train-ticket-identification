#ifndef _TEXT_BINARIZER_H_
#define _TEXT_BINARIZER_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/higuigui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::vector;
using cv::Mat;
using cv::Rect;
using cv::Point2i;
using cv::Size;

namespace yingjia{
	class TextBinarizer
	{
	public:
		TextBinarizer(int winSize1=3, int winSize2=75);
		~TextBinarizer();
		bool Run(Mat& src, Mat& dst);
		bool Run2(Mat& src, Mat& dst);
	private:
		bool ImgThreshold1(Mat& src, Mat& dst);
		bool ImgThreshold2(Mat& src, Mat& dst, int winSize);
		void CalcIntegralImg(Mat& src, Mat& dst);
		int EstimateStrokeWidth(Mat& imgCombine, Mat& imgSrc);
		void PostProcessing(Mat& src, Mat& dst);
		int contrastWinSize_;
		int localThreshWinSize_;
	};
}
© 2017 GitHub, Inc.
