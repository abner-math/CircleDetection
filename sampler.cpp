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
	for (size_t i = 0; i < numPoints(); i++)
	{
		mPoints[i] = i;
		if (++mNumPointsPerAngle[normalAngleIndex(i)] == 1)
			--mNumEmptyAngles;
	}
	for (size_t i = 0; i < HISTORY_SIZE; i++)
	{
		mLastPoints[i] = rand() % mPoints.size();
	}
	mLastPointIndex = 0;
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
{
	for (size_t i = 0; i < HISTORY_SIZE; i++)
	{
		mLastPoints[i] = rand() % mPoints.size();
	}
	mLastPointIndex = 0;
}

bool Sampler::canSample() const 
{
	return (numAngles() - mNumEmptyAngles) > mMinArcLength;
}

std::pair<size_t, size_t> Sampler::sample()  
{
	std::pair<size_t, size_t> p;
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.first = selectRandomPoint();
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeSample1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.second = selectRandomPoint();
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeSample2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return p;
}

void Sampler::removePoint(size_t point)
{
	if (isRemoved(point)) return;
	mPoints[point] = selectRandomPoint(point);
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

size_t Sampler::getValidPoint(const Quadtree *node, size_t angle) 
{
	if (!node->points(angle).empty()) return getPoint(node->randomPoint(angle));
	size_t clockwiseAngle = angle;
	size_t counterClockwiseAngle = angle;
	for (size_t i = 0; i < numAngles() / 2; i++)
	{
		clockwiseAngle = increaseOneAngle(clockwiseAngle);
		if (!node->points(clockwiseAngle).empty()) return getPoint(node->randomPoint(clockwiseAngle));
		counterClockwiseAngle = decreaseOneAngle(counterClockwiseAngle);
		if (!node->points(counterClockwiseAngle).empty()) return getPoint(node->randomPoint(counterClockwiseAngle));
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::selectRandomPoint(size_t point)
{
	size_t angle = rand() % numAngles();
	const Quadtree *node = mQuadtree->findLeaf(point);
	while (!node->isRoot() && getRandomNumber() < mClimbChance)
	{
		node = node->parent();
	}
	size_t anotherPoint;
	do
	{
		anotherPoint = getValidPoint(node, angle);
		if (anotherPoint != std::numeric_limits<size_t>::max()) return anotherPoint;
		node = node->parent();
	} while (!node->isRoot());
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::selectRandomPoint()  
{
	size_t point = selectRandomPoint(mLastPoints[rand() % HISTORY_SIZE]);
	mLastPoints[mLastPointIndex] = point;
	mLastPointIndex = (mLastPointIndex + 1) % HISTORY_SIZE;
	return point;
}
