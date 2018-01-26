#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ImageUtils
{
public:
	ImageUtils(const cv::Mat &img, int cannyLowThreshold, int cannyRatio = 3, int cannyKernelSize = 3);
	
	int index(int x, int y) const 
	{
		return y * mEdges.cols + x;
	}
	
	uchar edge(int index) const 
	{
		return mEdges.data[index];
	}
	
	float sobelAngle(int index) const 
	{
		return ((float*)mSobelAngle.data)[index];
	}
	
	cv::Point2f sobel(int index) const 
	{
		return cv::Point2f(((float*)mSobelX.data)[index], ((float*)mSobelY.data)[index]);
	}
	
	cv::Point2f inverseSobel(int index) const 
	{
		return cv::Point2f(((float*)mInverseSobelX.data)[index], ((float*)mInverseSobelY.data)[index]);
	}
	
	float curvature(int index);
	
private:
	cv::Mat mEdges;
	cv::Mat mSobelX;
	cv::Mat mSobelY;
	cv::Mat mInverseSobelX;
	cv::Mat mInverseSobelY;
	cv::Mat mSobelNorm;
	cv::Mat mSobelAngle;

};

#endif // IMAGEUTILS_H
