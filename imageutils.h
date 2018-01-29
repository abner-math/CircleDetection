#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ImageUtils
{
public:
	ImageUtils(const cv::Mat &img, short numAngles, int cannyLowThreshold, int cannyRatio = 3, int cannyKernelSize = 3);
	
	~ImageUtils();
	
	bool isEdge(int index) const 
	{
		return index >= 0 && index < mEdges.total() && mEdgeIndices[index] != -1;
	}
	
	int indexOf(int index) const 
	{
		return mEdgeIndices[index];
	}
	
	int numEdges() const 
	{
		return mNumEdges;
	}
	
	cv::Point2f position(int edgeIndex) const 
	{
		int index = mReverseEdgeIndices[edgeIndex];
		int x = index % mImg.cols;
		int y = index / mImg.cols;
		return cv::Point2f(x, y);
	}
	
	cv::Point2f sobel(int edgeIndex) const 
	{
		return cv::Point2f(((float*)mSobelX.data)[edgeIndex], ((float*)mSobelY.data)[edgeIndex]);
	}
	
	cv::Point2f inverseSobel(int edgeIndex) const 
	{
		return cv::Point2f(((float*)mInverseSobelX.data)[edgeIndex], ((float*)mInverseSobelY.data)[edgeIndex]);
	}
	
	float curvature(int edgeIndex);
	
	int createConnectedComponents();
	
	int labelOf(int edgeIndex)
	{
		return mLabels[edgeIndex];
	}
	
	int groupPointsByAngle();
	
	bool isCentroid(int edgeIndex)
	{
		return mGroups[edgeIndex] == edgeIndex;
	}
	
	short angleIndexOf(int edgeIndex)
	{
		return mAngleIndices[edgeIndex];
	}
	
private:
	cv::Mat mImg;
	cv::Mat mEdges;
	cv::Mat mSobelX;
	cv::Mat mSobelY;
	cv::Mat mInverseSobelX;
	cv::Mat mInverseSobelY;
	cv::Mat mSobelNorm;
	cv::Mat mSobelAngle;
	int mNumAngles;
	int mNumEdges;
	int *mEdgeIndices;
	int *mReverseEdgeIndices;
	float *mNeighborAngles;
	int *mLabels;
	short *mAngleIndices;
	int *mGroups;

	void createEdgeIndices();
	
	void createSobel();
	
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
	
	float sobelAngle(int index) const 
	{
		float angle = ((float*)mSobelAngle.data)[index];
		if (angle >= 180)
			return angle - 180;
		return angle;
	}
	
	void calculateAngleIndices();
	
};

#endif // IMAGEUTILS_H
