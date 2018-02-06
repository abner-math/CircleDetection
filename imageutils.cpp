#include "imageutils.h"

#include <iostream>

#include "ED.h"

ImageUtils::ImageUtils(const cv::Mat &img, short numAngles, int cannyLowThreshold, int cannyRatio, int cannyKernelSize)
	: mNumAngles(numAngles)
	, mEdgeIndices(NULL)
	, mNeighborAngles(NULL)
	, mLabels(NULL)
	, mAngleIndices(NULL)
	, mGroups(NULL)
	, mCountInGroup(NULL)
{	
	//EdgeMap *edgeMap = DetectEdgesByEDPF(img.data, img.cols, img.rows);
	//mEdges = cv::Mat(img.size(), CV_8U, edgeMap->edgeImg);
	cv::Canny(img, mEdges, cannyLowThreshold, cannyLowThreshold * cannyRatio, cannyKernelSize);
	createEdgeIndices();
	cv::GaussianBlur(img, mImg, cv::Size(3, 3), 0, 0);
	createNormals();
}

ImageUtils::~ImageUtils()
{
	delete[] mEdgeIndices;
	delete[] mReverseEdgeIndices;
	if (mNeighborAngles != NULL)
		delete[] mNeighborAngles;
	if (mLabels != NULL)
		delete[] mLabels;
	if (mAngleIndices != NULL)
		delete[] mAngleIndices;
	if (mGroups != NULL)
		delete[] mGroups;
	if (mCountInGroup != NULL)
		delete[] mCountInGroup;
}

void ImageUtils::createEdgeIndices()
{
	mNumEdges = 0;
	mEdgeIndices = new int[mEdges.total()];
	for (int index = 0; index < mEdges.total(); index++)
	{
		if (mEdges.data[index] > 0)
		{
			mEdgeIndices[index] = mNumEdges;
			++mNumEdges;
		}
		else
		{
			mEdgeIndices[index] = -1;
		}
	}
	mReverseEdgeIndices = new int[mNumEdges];
	for (int index = 0; index < mEdges.total(); index++)
	{
		if (mEdgeIndices[index] != -1)
		{
			mReverseEdgeIndices[mEdgeIndices[index]] = index;
		}
	}
}

void ImageUtils::createNormals()
{
	cv::Mat sobelX, sobelY;
	cv::Sobel(mImg, sobelX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(mImg, sobelY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	mSobelX = cv::Mat(mNumEdges, 1, CV_32F);
	mSobelY = cv::Mat(mNumEdges, 1, CV_32F);
	for (int index = 0; index < mNumEdges; index++)
	{
		((float*)mSobelX.data)[index] = ((float*)sobelX.data)[mReverseEdgeIndices[index]];
		((float*)mSobelY.data)[index] = ((float*)sobelY.data)[mReverseEdgeIndices[index]];
	}
	cv::magnitude(mSobelX, mSobelY, mSobelNorm);
	cv::divide(mSobelX, mSobelNorm, mSobelX);
	cv::divide(mSobelY, mSobelNorm, mSobelY);
	cv::phase(mSobelX, mSobelY, mSobelAngle, true);
}

void ImageUtils::calculateAngleIndices()
{
	mAngleIndices = new short[mNumEdges];
	for (int edgeIndex = 0; edgeIndex < mNumEdges; edgeIndex++)
	{
		mAngleIndices[edgeIndex] = (short)(std::round(normalAngle(edgeIndex) / (360.0f / mNumAngles))) % mNumAngles;
	}
}

void ImageUtils::calculateNeighborAngles()
{
	mNeighborAngles = new float[mNumEdges * 8];
	for (int edgeIndex = 0; edgeIndex < mNumEdges; edgeIndex++)
	{
		int index = mReverseEdgeIndices[edgeIndex];
		for (int i = 0; i < 8; i++)
		{
			int neighbor = neighborIndex(index, i);
			if (isEdge(neighbor))
			{
				int edgeNeighbor = mEdgeIndices[neighbor];
				if (i < 4)
				{
					mNeighborAngles[edgeIndex * 8 + i] = mNeighborAngles[edgeNeighbor * 8 + 7 - i];
				}
				else 
				{
					mNeighborAngles[edgeIndex * 8 + i] = angleBetween(edgeIndex, edgeNeighbor);
				}
			}
			else
			{
				mNeighborAngles[edgeIndex * 8 + i] = std::numeric_limits<float>::infinity();
			}
		}
	}
}
	
void ImageUtils::reorientNormals()
{
	for (int edgeSeed = 0; edgeSeed < mNumEdges; edgeSeed++)
	{
		int label = mLabels[edgeSeed];
		int index = mReverseEdgeIndices[edgeSeed];
		cv::Point2f position(index % mImg.cols, index / mImg.cols);
		if (normal(edgeSeed).dot(position - mCenters[label]) < 0)
		{
			((float*)mSobelX.data)[edgeSeed] = -((float*)mSobelX.data)[edgeSeed];
			((float*)mSobelY.data)[edgeSeed] = -((float*)mSobelY.data)[edgeSeed];
			((float*)mSobelAngle.data)[edgeSeed] = ((float*)mSobelAngle.data)[edgeSeed] + 180;
			if (((float*)mSobelAngle.data)[edgeSeed] > 360)
				((float*)mSobelAngle.data)[edgeSeed] = ((float*)mSobelAngle.data)[edgeSeed] - 360;
		}
	}
	
}

int ImageUtils::createConnectedComponents()
{
	calculateAngleIndices();
	calculateNeighborAngles();
	std::queue<int> queue;
	bool *marked = (bool*)calloc(mNumEdges, sizeof(bool));
	mLabels = new int[mNumEdges];
	int numLabels = 0;
	for (int edgeSeed = 0; edgeSeed < mNumEdges; edgeSeed++)
	{
		if (!marked[edgeSeed])
		{
			marked[edgeSeed] = true;
			mLabels[edgeSeed] = numLabels;
			queue.push(mReverseEdgeIndices[edgeSeed]);
			while (!queue.empty())
			{
				int index = queue.front();
				queue.pop();
				//int x = index % mImg.cols;
				//int y = index / mImg.cols;
				for (int i = 0; i < 8; i++)
				{
					int neighbor = neighborIndex(index, i);
					int edgeNeighbor = mEdgeIndices[neighbor];
					if (isEdge(neighbor) && !marked[edgeNeighbor] && mNeighborAngles[edgeNeighbor * 8 + 7 - i] < 6)
					{
						marked[edgeNeighbor] = true;
						mLabels[edgeNeighbor] = numLabels;
						queue.push(neighbor);
					}
				}
			}
			//mCenters.push_back(cv::Point2f((maxX + minX) / 2, (maxY + minY) / 2));
			++numLabels;
		}
	}
	delete[] marked;
	//reorientNormals();
	return numLabels;
}

int ImageUtils::groupPointsByAngle()
{
	mGroups = new int[mNumEdges];
	bool *marked = (bool*)calloc(mNumEdges, sizeof(bool));
	mCountInGroup = (int*)calloc(mNumEdges, sizeof(int));
	int numGroups = 0;
	std::queue<int> queue;
	for (int edgeSeed = 0; edgeSeed < mNumEdges; edgeSeed++)
	{
		int count = 0;
		if (!marked[edgeSeed])
		{
			marked[edgeSeed] = true;
			mGroups[edgeSeed] = edgeSeed;
			queue.push(mReverseEdgeIndices[edgeSeed]);
			while (!queue.empty())
			{
				int index = queue.front();
				queue.pop();
				++mCountInGroup[edgeSeed];
				for (int i = 0; i < 8; i++)
				{
					int neighbor = neighborIndex(index, i);
					int edgeNeighbor = mEdgeIndices[neighbor];
					if (isEdge(neighbor) && !marked[edgeNeighbor] && mLabels[edgeSeed] == mLabels[edgeNeighbor] && 
						(mAngleIndices[edgeSeed] == mAngleIndices[edgeNeighbor] || 
						mAngleIndices[edgeSeed] == (mAngleIndices[edgeNeighbor] + mNumAngles / 2) % mNumAngles))
					{
						marked[edgeNeighbor] = true;
						mGroups[edgeNeighbor] = edgeSeed;
						queue.push(neighbor);
					}
				}
			}
			++numGroups;
		}
	}
	delete[] marked;
	return numGroups;
}

/*
float ImageUtils::curvature(int edgeIndex)
{
	int index = mReverseEdgeIndices[edgeIndex];
	float sum = 0;
	int count = 0;
	for (int i = 0; i < 8; i++)
	{
		int neighbor = neighborIndex(index, i);
		if (isEdge(neighbor))
		{
			int edgeNeighbor = mEdgeIndices[neighbor];
			sum += mNeighborAngles[edgeNeighbor * 8 + 7 - i];
			++count;
		}
	}
	if (count == 0) return 0;
	return sum / count;
}
	*/
