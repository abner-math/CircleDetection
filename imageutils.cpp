#include "imageutils.h"

#include <iostream>

ImageUtils::ImageUtils(const cv::Mat &img, int cannyLowThreshold, int cannyRatio, int cannyKernelSize)
{	
	cv::Mat blur;
	cv::GaussianBlur(img, blur, cv::Size(5, 5), 0, 0);
	cv::Canny(blur, mEdges, cannyLowThreshold, cannyLowThreshold * cannyRatio, cannyKernelSize);
	cv::Sobel(blur, mSobelX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(blur, mSobelY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::magnitude(mSobelX, mSobelY, mSobelNorm);
	cv::divide(mSobelX, mSobelNorm, mSobelX);
	cv::divide(mSobelY, mSobelNorm, mSobelY);
	mInverseSobelX = 1 / mSobelX;
	mInverseSobelY = 1 / mSobelY;
	cv::phase(mSobelX, mSobelY, mSobelAngle, true);
}

float ImageUtils::curvature(int i)  
{
	int indices[8] = {
		i - mEdges.cols - 1,
		i - mEdges.cols,
		i - mEdges.cols + 1,
		i - 1,
		i + 1,
		i + mEdges.cols - 1,
		i + mEdges.cols,
		i + mEdges.cols + 1
	};
	float sum = 0;
	int count = 0; 
	cv::Point2f normal = sobel(i);
	for (size_t i = 0; i < 8; i++)
	{
		if (indices[i] >= 0 && indices[i] < mEdges.total() && edge(indices[i]))
		{
			sum += std::acos(normal.dot(sobel(indices[i])));
			++count;
		}
	}
	return sum / count;
}
