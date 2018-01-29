#include "quadtree.h"

#include <iostream>

Quadtree::Quadtree(const PointCloud *pointCloud, size_t minNumPoints, size_t minNumAngles, float minSize)
	: mPointCloud(pointCloud)
	, mMinNumPoints(minNumPoints)
	, mMinNumAngles(minNumAngles)
	, mMinSize(minSize)
	, mRoot(this)
	, mParent(NULL)
	, mChildren(std::vector<Quadtree*>(4, NULL))
	, mIsLeaf(true)
	, mPoints(std::vector<std::vector<size_t> >(pointCloud->numAngles()))
	, mNumPoints(pointCloud->numPoints())
	, mNumAngles(0)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	mCenter = pointCloud->extension().tl() + (pointCloud->extension().br() - pointCloud->extension().tl()) / 2;
	mSize = std::max(pointCloud->extension().width, pointCloud->extension().height) / 2;
	mLeaves = new std::vector<Quadtree*>(pointCloud->numPoints(), this);
	for (size_t i = 0; i < pointCloud->numAngles(); i++)
	{
		mPoints[i].reserve(pointCloud->numPoints());
	}
	for (size_t i = 0; i < pointCloud->numPoints(); i++)
	{
		size_t normalAngleIndex = pointCloud->point(i).angleIndex;
		if (mPoints[normalAngleIndex].empty())
			++mNumAngles;
		mPoints[normalAngleIndex].push_back(i);
	}
	build();
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeCreateQuadtree += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
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
		delete mLeaves;
}

Quadtree::Quadtree(Quadtree *parent)
	: mPointCloud(parent->mPointCloud)
	, mMinNumPoints(parent->mMinNumPoints)
	, mMinNumAngles(parent->mMinNumAngles)
	, mMinSize(parent->mMinSize)
	, mRoot(parent->mRoot)
	, mParent(parent)
	, mChildren(std::vector<Quadtree*>(4, NULL))
	, mIsLeaf(true)
	, mLeaves(NULL)
	, mPoints(std::vector<std::vector<size_t> >(parent->mPointCloud->numAngles()))
	, mNumPoints(0)
	, mNumAngles(0)
{
	mParent->mIsLeaf = false;
}

void Quadtree::build()
{
	if (mNumPoints <= mMinNumPoints || mNumAngles < mMinNumAngles || mSize < mMinSize) return;
	float halfSize = mSize / 2;
	float quarterSize = mSize / 4;
	for (size_t angle = 0; angle < mPointCloud->numAngles(); angle++)
	{
		for (const size_t &point : mPoints[angle])
		{
			size_t index = ((mPointCloud->point(point).position.x > mCenter.x) << 1) | (mPointCloud->point(point).position.y > mCenter.y);
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
			(*mRoot->mLeaves)[point] = mChildren[index];
		}
	}
	for (size_t i = 0; i < 4; i++)
	{
		if (mChildren[i] != NULL)
			mChildren[i]->build();
	}
}
