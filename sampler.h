#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include <random>
#include <set>

#include "quadtree.h"

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
	
	std::pair<size_t, size_t> sample(const std::set<size_t> &points) const;
	
	bool isRemoved(size_t point) const;
	
	void removePoint(size_t point);
	
	void removePointFromAll(size_t point);
	
private:
	static std::default_random_engine sRandomGenerator;
	static std::gamma_distribution<double> sGammaDistribution;
	static double sGammaThreshold;
	Sampler *mParent;
	std::vector<Sampler*> mChildren;
	const Quadtree *mQuadtree;
	const float mClimbChance;
	const size_t mMinArcLength;
	std::vector<size_t> mPoints;
	std::vector<size_t> mNumPointsPerAngle;
	size_t mNumEmptyAngles;
	size_t mNumAvailablePoints;
	
	inline size_t numAngles() const 
	{
		return mQuadtree->pointCloud()->numAngles();
	}
	
	inline size_t normalAngleIndex(size_t point) const 
	{
		return mQuadtree->pointCloud()->point(point)->normalAngleIndex;
	}
	
	size_t getPoint(size_t index) const;
	
	size_t selectRandomPoint(const std::set<size_t> &points) const;
	
	inline float getProbability() const 
	{
		return rand() / static_cast<float>(RAND_MAX);
	}
	
	bool containsValidPoints(const Quadtree *node, size_t firstPoint, size_t angle) const;
	
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
	
	size_t resolveAngle(const Quadtree *node, size_t firstPoint, size_t angle) const;
	
	size_t selectSubstitutePoint(size_t firstPoint) const;
	
	size_t selectAnotherRandomPoint(size_t firstPoint) const;
	
};

#endif // _SAMPLER_H_
