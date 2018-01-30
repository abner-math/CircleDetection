#ifndef _QUADTREE_H_
#define _QUADTREE_H_

#include "pointcloud.h"

class Quadtree
{
public:
	Quadtree(const PointCloud &pointCloud, float minSize);
	
	~Quadtree();
	
	const PointCloud& pointCloud() const 
	{
		return mPointCloud;
	}
	
	const Quadtree* parent() const 
	{
		return mParent;
	}
	
	const Quadtree* child(size_t index) const 
	{
		return mChildren[index];
	}
	
	const Quadtree* findLeaf(size_t pointIndex) const 
	{
		return mRoot->mLeaves[pointIndex];
	}
	
	bool isRoot() const 
	{
		return this == mRoot;
	}
	
	bool isLeaf() const 
	{
		return mIsLeaf;
	}
	
	const std::vector<size_t>& points(short angle) const 
	{
		return mPoints[angle];
	}
	
	cv::Rect2f rect() const 
	{
		return cv::Rect2f(mCenter.x - mSize / 2, mCenter.y - mSize / 2, mSize, mSize);
	}
	
	const cv::Point2f& center() const 
	{
		return mCenter;
	}
	
	float size() const 
	{
		return mSize;
	}
	
private:
	const PointCloud &mPointCloud;
	const float mMinSize;
	Quadtree *mRoot;
	Quadtree *mParent;
	Quadtree *mChildren[4];
	cv::Point2f mCenter;
	float mSize;
	bool mIsLeaf;
	std::vector<std::vector<size_t> > mPoints;
	size_t mNumPoints;
	short mNumAngles;
	Quadtree **mLeaves;
	
	Quadtree(Quadtree *parent);
	
	void build();
	
};

#endif // _QUADTREE_H_
