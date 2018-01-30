#include "sampler.h"

#include <iostream>

Sampler::Sampler(const PointCloud &pointCloud, float minQuadtreeSize, short minArcLength)
	: mPointCloud(pointCloud)
	, mMinArcLength(minArcLength)
	, mQuadtree(Quadtree(pointCloud, minQuadtreeSize))
	, mNumEmptyAngles(numAngles())
	, mNumAvailablePoints(numPoints())
{         
	mPoints = new size_t[numPoints()];
	mNumPointsPerAngle = (size_t*)calloc(numAngles(), sizeof(size_t));
	mRemovedPoints = new bool[numPoints()];
	mBlockedPoints = new bool[numPoints()];
	for (size_t i = 0; i < numPoints(); i++)
	{
		mPoints[i] = i;
		mRemovedPoints[i] = false;
		mBlockedPoints[i] = false;
		if (++mNumPointsPerAngle[angleIndex(i)] == 1)
			--mNumEmptyAngles;
	}
}

Sampler::~Sampler()
{
	delete[] mPoints;
	delete[] mNumPointsPerAngle;
	delete[] mRemovedPoints;
	delete[] mBlockedPoints;
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
	mRemovedPoints[point] = true;
	mPoints[point] = getSubstitutePoint(point);
	if (--mNumPointsPerAngle[angleIndex(point)] == 0)
		++mNumEmptyAngles;
	--mNumAvailablePoints;
}

void Sampler::blockPoint(size_t point, size_t newPoint)
{
	if (isBlocked(point)) return;
	mBlockedPoints[point] = true;
	mPoints[point] = newPoint;
	if (--mNumPointsPerAngle[angleIndex(point)] == 0)
		++mNumEmptyAngles;
	--mNumAvailablePoints;
}

void Sampler::unblockPoint(size_t point)
{
	if (!isBlocked(point)) return;
	mPoints[point] = point;
	if (mNumPointsPerAngle[angleIndex(point)]++ == 0)
		--mNumEmptyAngles;
	++mNumAvailablePoints;
	mBlockedPoints[point] = false;
}

size_t Sampler::getPoint(size_t index)  
{
	if (index == mPoints[index]) return index;
	mPoints[index] = getPoint(mPoints[index]);
	return mPoints[index];
}

size_t Sampler::getValidPoint(const std::vector<size_t> &points) const
{
	for (const size_t &point : points)
	{
		if (isAvailable(point))
		{
			return point;
		}
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::getValidPoint(const Quadtree *node, short angle) const
{
	size_t point = getValidPoint(node->points(angle));
	if (point < numPoints()) return point;
	short clockwiseAngle = angle;
	short counterClockwiseAngle = angle;
	for (short i = 0; i < numAngles() / 2; i++)
	{
		clockwiseAngle = increaseOneAngle(clockwiseAngle);
		point = getValidPoint(node->points(clockwiseAngle));
		if (point < numPoints()) return point;
		counterClockwiseAngle = decreaseOneAngle(counterClockwiseAngle);
		point = getValidPoint(node->points(counterClockwiseAngle));
		if (point < numPoints()) return point;
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::getSubstitutePoint(size_t point) 
{
	const Quadtree *node = mQuadtree.findLeaf(point);
	while (true)
	{
		size_t validPoint = getValidPoint(node, angleIndex(point));
		if (validPoint < numPoints()) return validPoint;
		if (node->isRoot()) break;
		node = node->parent();
	}
	//std::cout << "Ran out of valid points" << std::endl;
	return getPoint(rand() % numPoints());
}

size_t Sampler::selectRandomPoint()  
{
	return getPoint(rand() % numPoints());
}
