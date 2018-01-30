#include "quadtree.h"

#include <iostream>

Quadtree::Quadtree(const PointCloud &pointCloud, float minSize)
	: mPointCloud(pointCloud)
	, mMinSize(minSize)
	, mRoot(this)
	, mParent(NULL)
	, mIsLeaf(true)
	, mPoints(std::vector<std::vector<size_t> >(pointCloud.numAngles()))
	, mNumPoints(pointCloud.numPoints())
	, mNumAngles(0)
{
	mCenter = pointCloud.extension().tl() + (pointCloud.extension().br() - pointCloud.extension().tl()) / 2;
	mSize = std::max(pointCloud.extension().width, pointCloud.extension().height);
	mLeaves = new Quadtree*[pointCloud.numGroups()];
	for (short i = 0; i < pointCloud.numAngles(); i++)
	{
		mPoints[i].reserve(pointCloud.numGroups());
	}
	for (size_t i = 0; i < pointCloud.numGroups(); i++)
	{
		short angleIndex = pointCloud.group(i).angleIndex;
		if (mPoints[angleIndex].empty())
			++mNumAngles;
		mPoints[angleIndex].push_back(i);
		mLeaves[i] = this;
	}
	for (size_t i = 0; i < 4; i++)
	{
		mChildren[i] = NULL;
	}
	build();
}

Quadtree::~Quadtree()
{
	for (size_t i = 0; i < 4; i++)
	{
		if (mChildren[i] != NULL)
		{
			delete mChildren[i];
		}
	}
	if (mLeaves != NULL)
		delete[] mLeaves;
}

Quadtree::Quadtree(Quadtree *parent)
	: mPointCloud(parent->mPointCloud)
	, mMinSize(parent->mMinSize)
	, mRoot(parent->mRoot)
	, mParent(parent)
	, mIsLeaf(true)
	, mLeaves(NULL)
	, mPoints(std::vector<std::vector<size_t> >(parent->mPointCloud.numAngles()))
	, mNumPoints(0)
	, mNumAngles(0)
{
	mParent->mIsLeaf = false;
	for (size_t i = 0; i < 4; i++)
	{
		mChildren[i] = NULL;
	}
}

void Quadtree::build()
{
	if (mNumPoints < 10 || mNumAngles < 2 || mSize < mMinSize) return;
	float halfSize = mSize / 2;
	float quarterSize = mSize / 4;
	for (short angle = 0; angle < mPointCloud.numAngles(); angle++)
	{
		for (const size_t &point : mPoints[angle])
		{
			size_t index = ((mPointCloud.group(point).position.x > mCenter.x) << 1) | (mPointCloud.group(point).position.y > mCenter.y);
			if (mChildren[index] == NULL)
			{
				mChildren[index] = new Quadtree(this);
				mChildren[index]->mSize = halfSize;
				mChildren[index]->mCenter = mCenter + cv::Point2f(quarterSize * ((index >> 1) * 2.0f - 1.0f), quarterSize * ((index & 1) * 2.0f - 1.0f));
				mChildren[index]->mPoints[angle].reserve(mPoints[angle].size());
			}
			if (mChildren[index]->mPoints[angle].empty())
				++mChildren[index]->mNumAngles;
			mChildren[index]->mPoints[angle].push_back(point);
			++mChildren[index]->mNumPoints;
			mRoot->mLeaves[point] = mChildren[index];
		}
	}
	for (size_t i = 0; i < 4; i++)
	{
		if (mChildren[i] != NULL)
			mChildren[i]->build();
	}
}
