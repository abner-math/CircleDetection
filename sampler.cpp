#include "sampler.h"

#include <iostream>

boost::mt19937 Sampler::sRNG;
boost::exponential_distribution<float> Sampler::sDistribution = boost::exponential_distribution<float>(3.5f);
boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> > Sampler::sGenerator = boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> >(sRNG, sDistribution);
	
Sampler::Sampler(const PointCloud &pointCloud, short minArcLength)
	: mPointCloud(pointCloud)
	, mMinArcLength(minArcLength)
	, mNumEmptyAngles(numAngles())
	, mNumAvailablePoints(numPoints())
{         
	mPoints = new size_t[numPoints()];
	mNumPointsPerAngle = (size_t*)calloc(numAngles(), sizeof(size_t));
	mTotalNumPointsPerAngle = (size_t*)calloc(numAngles(), sizeof(size_t));
	mNumPicksPerPoint = new size_t[numPoints()];
	for (size_t i = 0; i < numPoints(); i++)
	{
		mPoints[i] = i;
		mNumPicksPerPoint[i] = 0;
		++mTotalNumPointsPerAngle[angleIndex(i)];
		if (++mNumPointsPerAngle[angleIndex(i)] == 1)
			--mNumEmptyAngles;
	}
	mPointsPerAngle = new size_t*[numAngles()];
	std::vector<int> indices(numAngles());
	for (short i = 0; i < numAngles(); i++)
	{
		indices[i] = 0;
		mPointsPerAngle[i] = new size_t[mTotalNumPointsPerAngle[i]];
	}
	for (size_t i = 0; i < numPoints(); i++)
	{
		short angle = angleIndex(i);
		mPointsPerAngle[angle][indices[angle]] = i;
		++indices[angle];
	}
}

Sampler::~Sampler()
{
	delete[] mPoints;
	delete[] mNumPointsPerAngle;
	delete[] mTotalNumPointsPerAngle;
	delete[] mNumPicksPerPoint;
	for (short i = 0; i < numAngles(); i++)
	{
		delete[] mPointsPerAngle[i];
	}
	delete[] mPointsPerAngle;
}

bool Sampler::canSample() const 
{
	return mPointCloud.numGroups() > mMinArcLength && mNumAvailablePoints > 2 * mMinArcLength && (numAngles() - mNumEmptyAngles) > mMinArcLength;
}

std::pair<size_t, size_t> Sampler::sample()  
{
	std::pair<size_t, size_t> p;
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.first = selectRandomPoint();
	if (++mNumPicksPerPoint[p.first] > 10)
	{
		removePoint(p.first);
	}
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeSample1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.second = selectAnotherRandomPoint(p.first);
	if (++mNumPicksPerPoint[p.second] > 10)
	{
		removePoint(p.second);
	}
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeSample2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return p;
}

void Sampler::removePoint(size_t point)
{
	if (isRemoved(point)) return;
	mPoints[point] = getPoint(rand() % numPoints());
	if (--mNumPointsPerAngle[angleIndex(point)] == 0)
		++mNumEmptyAngles;
	--mNumAvailablePoints;
}

size_t Sampler::getPoint(size_t index)  
{
	if (index == mPoints[index]) return index;
	mPoints[index] = getPoint(mPoints[index]);
	return mPoints[index];
}

size_t Sampler::selectRandomPointWithAngle(short angle) 
{
	size_t n = mTotalNumPointsPerAngle[angle];
	if (n == 0) return std::numeric_limits<size_t>::max();
	size_t point = (size_t)(getRandomValueFromExponentialDist() * n);
	return getPoint(mPointsPerAngle[angle][point]);
}

size_t Sampler::selectRandomPointWithValidAngle(short angle) 
{
	size_t point = selectRandomPointWithAngle(angle);
	if (point < numPoints()) return point;
	short clockwiseAngle = angle;
	short counterClockwiseAngle = angle;
	for (short i = 0; i < numAngles() / 2; i++)
	{
		clockwiseAngle = increaseOneAngle(clockwiseAngle);
		point = selectRandomPointWithAngle(clockwiseAngle);
		if (point < numPoints()) return point;
		counterClockwiseAngle = decreaseOneAngle(counterClockwiseAngle);
		point = selectRandomPointWithAngle(counterClockwiseAngle);
		if (point < numPoints()) return point;
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::selectRandomPoint()  
{
	size_t point = (size_t)(getRandomValueFromExponentialDist() * numPoints());
	return getPoint(point);
}

size_t Sampler::selectAnotherRandomPoint(size_t point)
{
	short angle = (short)(getRandomValueFromExponentialDist() * (numAngles() / 2) + angleIndex(point) + numAngles() / 4) % numAngles();
	size_t anotherPoint = selectRandomPointWithValidAngle(angle);
	if (anotherPoint < numPoints()) return anotherPoint;
	return selectRandomPoint();
}
	
