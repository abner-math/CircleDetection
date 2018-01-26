#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include <random>
#include <set>

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

#include "quadtree.h"

typedef boost::variate_generator< boost::mt19937, boost::binomial_distribution<> > GENERATOR_TYPE;

class Sampler
{
public:
	Sampler(const Quadtree *quadtree, float climbChance, size_t minArcLength);
	
	Sampler(Sampler &sampler);
	
	~Sampler();
	
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
	
	std::pair<size_t, size_t> sample(const std::vector<size_t> &points);
	
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
	GENERATOR_TYPE *mRandomPointGenerator;
	GENERATOR_TYPE *mRandomAngleGenerator;
	
	inline size_t numAngles() const 
	{
		return mQuadtree->pointCloud()->numAngles();
	}
	
	inline size_t normalAngleIndex(size_t point) const 
	{
		return mQuadtree->pointCloud()->point(point).normalAngleIndex;
	}
	
	size_t getPoint(size_t index);
	
	size_t selectRandomPoint(const std::vector<size_t> &points);
	
	inline float getRandomNumber() const
	{
		return rand() / static_cast<float>(RAND_MAX);
	}
	
	inline size_t getRandomPoint() const 
	{
		return (*mRandomPointGenerator)();
	}
	
	inline size_t getRandomAngle() const 
	{
		return (*mRandomAngleGenerator)();
	}
	
	size_t getValidPoint(const Quadtree *node, size_t firstPoint, size_t angle);
	
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
	
	inline size_t selectOppositeAngle(size_t angle) const
	{
		size_t x = (getRandomAngle() + angle + numAngles() / 2) % numAngles();
		return x;
	}
	
	size_t getNextValidPoint(const Quadtree *node, size_t firstPoint, size_t angle);
	
	size_t selectAnotherRandomPoint(size_t firstPoint);
	
};

#endif // _SAMPLER_H_
