#include "sampler.h"

#include <iostream>

Sampler::Sampler(const Quadtree *quadtree, float climbChance, size_t minArcLength)
	: mQuadtree(quadtree)
	, mClimbChance(climbChance)
	, mMinArcLength(minArcLength)
	, mParent(this)
	, mRoot(this)
	, mPoints(std::vector<size_t>(numPoints()))
	, mNumPointsPerAngle(std::vector<size_t>(numAngles(), 0))
	, mNumEmptyAngles(numAngles())
	, mNumAvailablePoints(numPoints())
{
	boost::mt19937 rng;
	boost::binomial_distribution<> pointDistribution(numPoints(), 0.1);
	mRandomPointGenerator = new GENERATOR_TYPE(rng, pointDistribution);
	boost::binomial_distribution<> angleDistribution(numAngles(), 0.1);
	mRandomAngleGenerator = new GENERATOR_TYPE(rng, angleDistribution);            
	for (size_t i = 0; i < numPoints(); i++)
	{
		mPoints[i] = i;
		if (++mNumPointsPerAngle[normalAngleIndex(i)] == 1)
			--mNumEmptyAngles;
	}
}

Sampler::Sampler(Sampler &parent)
	: mQuadtree(parent.mQuadtree)
	, mClimbChance(parent.mClimbChance)
	, mMinArcLength(parent.mMinArcLength)
	, mParent(&parent)
	, mRoot(parent.mRoot)
	, mPoints(parent.mPoints)
	, mNumPointsPerAngle(parent.mNumPointsPerAngle)
	, mNumEmptyAngles(parent.mNumEmptyAngles)
	, mNumAvailablePoints(parent.mNumAvailablePoints)
	, mRandomPointGenerator(parent.mRandomPointGenerator)
	, mRandomAngleGenerator(parent.mRandomAngleGenerator)
{
	
}

Sampler::~Sampler() 
{
	if (mParent == this)
	{
		delete mRandomPointGenerator;
		delete mRandomAngleGenerator;
	}
}

bool Sampler::canSample() const 
{
	return (numAngles() - mNumEmptyAngles) > mMinArcLength;
}

std::pair<size_t, size_t> Sampler::sample(const std::vector<size_t> &points)  
{
	std::pair<size_t, size_t> p;
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.first = selectRandomPoint(points);
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeSample1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.second = selectAnotherRandomPoint(p.first);
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeSample2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return p;
}

void Sampler::removePoint(size_t point)
{
	if (isRemoved(point)) return;
	mPoints[point] = selectAnotherRandomPoint(point);
	if (--mNumPointsPerAngle[normalAngleIndex(point)] == 0)
		++mNumEmptyAngles;
	--mNumAvailablePoints;
}

void Sampler::removePointFromAll(size_t point)
{
	removePoint(point);
	if (mParent != this)
		mParent->removePointFromAll(point);
}

size_t Sampler::getPoint(size_t index)  
{
	if (index == mPoints[index]) return index;
	mPoints[index] = getPoint(mPoints[index]);
	return mPoints[index];
}

size_t Sampler::selectRandomPoint(const std::vector<size_t> &points)  
{
	if (points.empty())
	{
		return getPoint(getRandomPoint());
	}
	else
	{
		size_t point = points[rand() % points.size()];
		size_t angle = normalAngleIndex(point);
		const Quadtree *node = mQuadtree->findLeaf(point);
		while (!node->isRoot() && getRandomNumber() < mClimbChance)
		{
			node = node->parent();
		}
		return getPoint(node->points(angle)[rand() % node->points(angle).size()]);
	}
}

size_t Sampler::getValidPoint(const Quadtree *node, size_t firstPoint, size_t angle) 
{
	/*for (const size_t &point : node->points(angle))
	{
		if (point != firstPoint && !isRemoved(point))
			return point;
	}*/
	if (!node->points(angle).empty())
	{
		return getPoint(node->points(angle)[rand() % node->points(angle).size()]);
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::getNextValidPoint(const Quadtree *node, size_t firstPoint, size_t angle) 
{
	size_t point = getValidPoint(node, firstPoint, angle);
	if (point != std::numeric_limits<size_t>::max()) return point;
	size_t clockwiseAngle = angle;
	size_t counterClockwiseAngle = angle;
	for (size_t i = 0; i < numAngles() / 2; i++)
	{
		clockwiseAngle = increaseOneAngle(clockwiseAngle);
		point = getValidPoint(node, firstPoint, clockwiseAngle);
		if (point != std::numeric_limits<size_t>::max()) return point;
		counterClockwiseAngle = decreaseOneAngle(counterClockwiseAngle);
		point = getValidPoint(node, firstPoint, counterClockwiseAngle);
		if (point != std::numeric_limits<size_t>::max()) return point;
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::selectAnotherRandomPoint(size_t point)  
{
	const Quadtree *node = mQuadtree->findLeaf(point); 
	while (!node->isRoot() && getRandomNumber() < mClimbChance)
	{
		node = node->parent();
	}
	size_t angle = selectOppositeAngle(normalAngleIndex(point));
	size_t anotherPoint;
	do
	{
		#ifdef _BENCHMARK
			auto begin = std::chrono::high_resolution_clock::now();
		#endif 
		anotherPoint = getNextValidPoint(node, point, angle);
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeDebug += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif
		if (anotherPoint != std::numeric_limits<size_t>::max())
		{
			return anotherPoint;
		}
		node = node->parent();
	} while (!node->isRoot());
	return getPoint(getRandomPoint());
}
