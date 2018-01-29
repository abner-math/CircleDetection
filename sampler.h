#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include <random>
#include <set>

#include "quadtree.h"

#define HISTORY_SIZE 10

class Sampler
{
public:
	Sampler(const Quadtree *quadtree, float climbChance, size_t minArcLength);
	
	Sampler(Sampler &sampler);
	
	size_t numAvailablePoints() const 
	{
		return mNumAvailablePoints;
	}
	
	size_t numPoints() const 
	{
		return mQuadtree->pointCloud()->numPoints();
	}
	
	const Quadtree* quadtree() const 
	{
		return mQuadtree;
	}
	
	float climbChance() const 
	{
		return mClimbChance;
	}
	
	size_t minArcLength() const 
	{
		return mMinArcLength;
	}
	
	bool canSample() const;
	
	std::pair<size_t, size_t> sample();
	
	bool isRemoved(size_t point) const
	{
		return mPoints[point] != point;
	}
	
	void removePoint(size_t point);
	
	void removePointFromAll(size_t point);
	
private:
	const Quadtree *mQuadtree;
	const float mClimbChance;
	const size_t mMinArcLength;
	Sampler *mRoot;
	Sampler *mParent;
	std::vector<size_t> mPoints;
	std::vector<size_t> mNumPointsPerAngle;
	size_t mNumEmptyAngles;
	size_t mNumAvailablePoints;
	size_t mLastPoints[HISTORY_SIZE];
	size_t mLastPointIndex;
	
	inline size_t numAngles() const 
	{
		return mQuadtree->pointCloud()->numAngles();
	}
	
	inline size_t normalAngleIndex(size_t point) const 
	{
		return 0;//mQuadtree->pointCloud()->point(point).normalAngleIndex;
	}
	
	inline float getRandomNumber() const
	{
		return rand() / static_cast<float>(RAND_MAX);
	}
	
	inline size_t decreaseOneAngle(size_t angle) const
	{
		if (angle == 0) return numAngles() - 1;
		return angle - 1;
	}
	
	inline size_t increaseOneAngle(size_t angle) const 
	{
		if (angle == numAngles() - 1) return 0;
		return angle + 1;
	}
	
	size_t getPoint(size_t index);
	
	size_t getValidPoint(const Quadtree *node, size_t angle);
	
	size_t selectRandomPoint(size_t point);
	
	size_t selectRandomPoint();
	
};

#endif // _SAMPLER_H_
