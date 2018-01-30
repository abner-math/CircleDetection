#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include <random>
#include <set>

#include "quadtree.h"

class Sampler
{
public:
	Sampler(const PointCloud &pointCloud, float minQuadtreeSize, short minArcLength);
	
	~Sampler();
	
	size_t numAvailablePoints() const 
	{
		return mNumAvailablePoints;
	}
	
	size_t numPoints() const 
	{
		return mPointCloud.numGroups();
	}
	
	short numAngles() const 
	{
		return mPointCloud.numAngles();
	}
	
	const PointCloud& pointCloud() const 
	{
		return mPointCloud;
	}
	
	const Quadtree& quadtree() const 
	{
		return mQuadtree;
	}
	
	short minArcLength() const 
	{
		return mMinArcLength;
	}
	
	bool canSample() const;
	
	std::pair<size_t, size_t> sample();
	
	void removePoint(size_t point);
	
	bool isRemoved(size_t point) const
	{
		return mRemovedPoints[point];
	}
	
	void blockPoint(size_t point);
	
	void unblockPoint(size_t point);
	
	bool isBlocked(size_t point) const 
	{
		return mBlockedPoints[point];
	}
	
	bool isAvailable(size_t point) const 
	{
		return !isRemoved(point) && !isBlocked(point);
	}
	
private:
	const PointCloud &mPointCloud;
	const short mMinArcLength;
	Quadtree mQuadtree;
	size_t *mPoints;
	size_t *mNumPointsPerAngle;
	bool *mRemovedPoints;
	bool *mBlockedPoints;
	size_t mNumEmptyAngles;
	size_t mNumAvailablePoints;
	
	short angleIndex(size_t point) const 
	{
		return mPointCloud.group(point).angleIndex;
	}
	
	size_t getPoint(size_t index);
	
	short decreaseOneAngle(short angle) const
	{
		if (angle == 0) return numAngles() - 1;
		return angle - 1;
	}
	
	short increaseOneAngle(short angle) const 
	{
		if (angle == numAngles() - 1) return 0;
		return angle + 1;
	}
	
	size_t getValidPoint(const std::vector<size_t> &points) const;
	
	size_t getValidPoint(const Quadtree *node, short angle) const;
	
	size_t getSubstitutePoint(size_t point);
	
	size_t selectRandomPoint();
	
};

#endif // _SAMPLER_H_
