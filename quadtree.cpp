#include "quadtree.h"

Quadtree::Quadtree(const PointCloud *pointCloud, size_t minNumPoints)
	: mPointCloud(pointCloud)
	, mMinNumPoints(minNumPoints)
	, mRoot(this)
	, mParent(NULL)
	, mIsLeaf(true)
	, mPoints(std::vector<std::vector<size_t> >(pointCloud->numAngles()))
	, mNumPoints(pointCloud->numPoints())
{
	for (size_t i = 0; i < 4; i++)
		mChildren[i] = NULL;
	mCenter = pointCloud->extension().tl() + (pointCloud->extension().br() - pointCloud->extension().tl()) / 2;
	mSize = std::max(pointCloud->extension().width, pointCloud->extension().height) / 2;
	mLeaves = new std::vector<Quadtree*>(pointCloud->numPoints(), this);
	for (size_t i = 0; i < pointCloud->numAngles(); i++)
	{
		mPoints[i].reserve(pointCloud->numPoints());
	}
	for (size_t i = 0; i < pointCloud->numPoints(); i++)
	{
		mPoints[pointCloud->point(i)->normalAngleIndex].push_back(i);
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
		delete mLeaves;
}

Quadtree::Quadtree(Quadtree *parent)
	: mPointCloud(parent->mPointCloud)
	, mRoot(parent->mRoot)
	, mParent(parent)
	, mIsLeaf(true)
	, mLeaves(NULL)
	, mPoints(std::vector<std::vector<size_t> >(parent->mPointCloud->numAngles()))
	, mMinNumPoints(parent->mMinNumPoints)
	, mNumPoints(0)
{
	for (size_t i = 0; i < 4; i++)
		mChildren[i] = NULL;
	mParent->mIsLeaf = false;
}

void Quadtree::build()
{
	if (mNumPoints <= mMinNumPoints) return;
	float halfSize = mSize / 2;
	float quarterSize = mSize / 4;
	for (size_t angle = 0; angle < mPointCloud->numAngles(); angle++)
	{
		for (const size_t &point : mPoints[angle])
		{
			size_t index = ((mPointCloud->point(point)->position.x > mCenter.x) << 1) | (mPointCloud->point(point)->position.y > mCenter.y);
			if (mChildren[index] == NULL)
			{
				mChildren[index] = new Quadtree(this);
				mChildren[index]->mSize = halfSize;
				mChildren[index]->mCenter = mCenter + cv::Point2f(quarterSize * ((index >> 1) * 2.0f - 1.0f), quarterSize * ((index & 1) * 2.0f - 1.0f));
				mChildren[index]->mPoints[angle].reserve(mPoints[angle].size());
			}
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
