#include "imageutils.h"

#include <iostream>

#include "ED.h"

ImageUtils::ImageUtils(const cv::Mat &img, short numAngles, int cannyLowThreshold, int cannyRatio, int cannyKernelSize)
	: mImg(img)
	, mNumAngles(numAngles)
	, mEdgeIndices(NULL)
	, mNeighborAngles(NULL)
	, mLabels(NULL)
	, mAngleIndices(NULL)
{	
	/*cv::Mat blurred; double sigma = 1, threshold = 5, amount = 1;
	cv::GaussianBlur(mImg, blurred, cv::Size(), sigma, sigma);
	cv::Mat lowContrastMask = abs(mImg - blurred) < threshold;
	cv::Mat sharpened = mImg*(1+amount) + blurred*(-amount);
	mImg.copyTo(sharpened, lowContrastMask);
	mImg = sharpened;*/

	//EdgeMap *edgeMap = DetectEdgesByEDPF(img.data, img.cols, img.rows);
	//mEdges = cv::Mat(img.size(), CV_8U, edgeMap->edgeImg);
	cv::Canny(mImg, mEdges, cannyLowThreshold, cannyLowThreshold * cannyRatio, cannyKernelSize);
	createEdgeIndices();
	//cv::GaussianBlur(mImg, mImg, cv::Size(5, 5), 0, 0);

	cv::Mat blur;
	cv::GaussianBlur(mImg, blur, cv::Size(0, 0), 3);
	cv::addWeighted(mImg, 1.5, blur, -0.5, 0, blur);
	createNormals();
	calculateAngleIndices();
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
	short anglesPerGroup = (180 / mNumAngles);
	for (int edgeIndex = 0; edgeIndex < mNumEdges; edgeIndex++)
	{
		mAngleIndices[edgeIndex] = (normalAngle(edgeIndex) % 180) / anglesPerGroup;
	}
}

void ImageUtils::calculateNeighborAngles()
{
	mNeighborAngles = new short[mNumEdges * 8];
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
		}
	}
}
	
int ImageUtils::createConnectedComponents()
{
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
			size_t countInLabel = 0;
			while (!queue.empty())
			{
				int index = queue.front();
				queue.pop();
				++countInLabel;
				for (int i = 0; i < 8; i++)
				{
					int neighbor = neighborIndex(index, i);
					int edgeNeighbor = mEdgeIndices[neighbor];
					if (isEdge(neighbor) && !marked[edgeNeighbor] && mNeighborAngles[edgeNeighbor * 8 + 7 - i] < 20)
					{
						marked[edgeNeighbor] = true;
						mLabels[edgeNeighbor] = numLabels;
						queue.push(neighbor);
					}
				}
			}
			++numLabels;
			mLabelsCount.push_back(countInLabel);
		}
	}
	delete[] marked;
	return numLabels;
}

float ImageUtils::curvature(int edgeIndex) const
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
			short angle = mNeighborAngles[edgeNeighbor * 8 + 7 - i];
			sum += angle;
			++count;
		}
	}
	if (count == 0) return 0;
	return sum / count;
}
	
