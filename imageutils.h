#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ImageUtils
{
public:
	ImageUtils(const cv::Mat &img, int cannyLowThreshold, int cannyRatio = 3, int cannyKernelSize = 3);
	
	~ImageUtils();
	
	bool isEdge(int index) const 
	{
		return mIsEdges[index];
	}
	
	float sobelAngle(int index) const 
	{
		float angle = ((float*)mSobelAngle.data)[index];
		if (angle >= 180)
			return angle - 180;
		return angle;
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
	
	int createConnectedComponents();
	
	int labelOf(int index)
	{
		return mLabels[index];
	}
	
private:
	cv::Mat mEdges;
	cv::Mat mSobelX;
	cv::Mat mSobelY;
	cv::Mat mInverseSobelX;
	cv::Mat mInverseSobelY;
	cv::Mat mSobelNorm;
	cv::Mat mSobelAngle;
	float *mNeighborAngles;
	bool *mIsEdges;
	int *mLabels;

	bool isValid(int index) const 
	{
		return index >= 0 && index < mEdges.total() && mEdges.data[index] > 0;
	}
	
	float angleBetween(int a, int b) const 
	{
		return std::acos(std::abs(sobel(a).dot(sobel(b)))) * 180.0f / M_PI;
	}
	
	int neighborIndex(int index, int neighbor) const 
	{
		switch (neighbor)
		{
			case 0:
				return index - mEdges.cols - 1;
			case 1:
				return index - mEdges.cols;
			case 2:
				return index - mEdges.cols + 1;
			case 3:
				return index - 1;
			case 4:
				return index + 1;
			case 5:
				return index + mEdges.cols - 1;
			case 6:
				return index + mEdges.cols;
			case 7:
				return index + mEdges.cols + 1;
			default:
				return index;
		}
	}
	
	void calculateNeighborAngles();
	
};

#endif // IMAGEUTILS_H
