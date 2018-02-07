#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

inline float norm(const cv::Point2f &p)
{
	return std::sqrt(p.x * p.x + p.y * p.y);
}

class ImageUtils
{
public:
	ImageUtils(const cv::Mat &img, short numAngles, int cannyLowThreshold, int cannyRatio = 3, int cannyKernelSize = 3);
	
	~ImageUtils();
	
	const cv::Mat& edgeImg() const 
	{
		return mEdges;
	}
	
	bool isEdge(int index) const 
	{
		return index >= 0 && index < mEdges.total() && mEdgeIndices[index] != -1;
	}
	
	int indexOf(int index) const 
	{
		return mEdgeIndices[index];
	}
	
	int reverseIndexOf(int edgeIndex) const 
	{
		return mReverseEdgeIndices[edgeIndex];
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
	
	cv::Point2f normal(int edgeIndex) const 
	{
		return cv::Point2f(((float*)mSobelX.data)[edgeIndex], ((float*)mSobelY.data)[edgeIndex]);
	}
	
	short normalAngle(int edgeIndex) const 
	{
		return static_cast<short>(((float*)mSobelAngle.data)[edgeIndex]);
	}
	
	short angleIndexOf(int edgeIndex)
	{
		return mAngleIndices[edgeIndex];
	}
	
	int createConnectedComponents();
	
	int labelOf(int edgeIndex)
	{
		return mLabels[edgeIndex];
	}
	
	size_t countPointsInComponent(int label)
	{
		return mLabelsCount[label];
	}
	
	int groupPointsByAngle();
	
	int groupOf(int edgeIndex)
	{
		return mGroups[edgeIndex];
	}
	
	float curvature(int edgeIndex) const;
	
private:
	cv::Mat mImg;
	cv::Mat mEdges;
	cv::Mat mSobelX;
	cv::Mat mSobelY;
	cv::Mat mSobelNorm;
	cv::Mat mSobelAngle;
	short mNumAngles;
	int mNumEdges;
	int *mEdgeIndices;
	int *mReverseEdgeIndices;
	short *mNeighborAngles;
	int *mLabels;
	std::vector<size_t> mLabelsCount;
	short *mAngleIndices;
	int *mGroups;

	void createEdgeIndices();
	
	void createNormals();
	
	short angleBetween(int a, int b) const 
	{
		short diff = std::abs((normalAngle(a) % 180) - (normalAngle(b) % 180));
		return std::min(diff, (short)(180 - diff));
		//return std::acos(std::abs(normal(a).dot(normal(b)))) * 180.0f / M_PI;
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
	
	void calculateAngleIndices();
	
};

#endif // IMAGEUTILS_H
