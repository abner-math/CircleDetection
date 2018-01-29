#include "imageutils.h"

#include <iostream>

ImageUtils::ImageUtils(const cv::Mat &img, int cannyLowThreshold, int cannyRatio, int cannyKernelSize)
	: mNeighborAngles(new float[img.total() * 8])
	, mIsEdges(new bool[img.total()])
	, mLabels(NULL)
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
	calculateNeighborAngles();
}

ImageUtils::~ImageUtils()
{
	delete[] mNeighborAngles;
	delete[] mIsEdges;
	if (mLabels != NULL)
		delete[] mLabels;
}

void ImageUtils::calculateNeighborAngles()
{
	for (int index = 0; index < mEdges.total(); index++)
	{
		mIsEdges[index] = mEdges.data[index] > 0;
		if (mIsEdges[index])
		{
			for (int i = 0; i < 8; i++)
			{
				int neighbor = neighborIndex(index, i);
				if (isValid(neighbor))
				{
					//if (i < 4)
					//{
					//	mNeighborAngles[index * 8 + i] = mNeighborAngles[neighbor * 8 + 7 - i];
					//}
					//else 
					//{
						mNeighborAngles[index * 8 + i] = angleBetween(index, neighbor);
					//}
				}
				else
				{
					mNeighborAngles[index * 8 + i] = std::numeric_limits<float>::infinity();
				}
			}
		}
	}
}
	
float ImageUtils::curvature(int index)  
{
	float sum = 0;
	int count = 0; 
	for (size_t i = 0; i < 8; i++)
	{
		float angle = mNeighborAngles[index * 8 + i];
		if (!std::isinf(angle))
		{
			sum += angle;
			++count;
		}
	}
	if (count == 0) return 0;
	return sum / count;
}

int ImageUtils::createConnectedComponents()
{
	std::queue<int> queue;
	bool *marked = (bool*)calloc(mEdges.total(), sizeof(bool));
	mLabels = new int[mEdges.total()];
	int numLabels = 0;
	for (int index = 0; index < mEdges.total(); index++)
	{
		if (isEdge(index) && !marked[index])
		{
			marked[index] = true;
			mLabels[index] = numLabels;
			queue.push(index);
			while (!queue.empty())
			{
				int index = queue.front();
				queue.pop();
				for (int i = 0; i < 8; i++)
				{
					int neighbor = neighborIndex(index, i);
					if (isValid(neighbor) && !marked[neighbor] && mNeighborAngles[index * 8 + i] < 20)
					{
						marked[neighbor] = true;
						mLabels[neighbor] = numLabels;
						queue.push(neighbor);
					}
				}
			}
			++numLabels;
		}
	}
	delete[] marked;
	return numLabels;
}
	
