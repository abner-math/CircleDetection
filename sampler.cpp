#include "sampler.h"

#include <iostream>

boost::mt19937 Sampler::sRNG;
boost::exponential_distribution<float> Sampler::sDistribution = boost::exponential_distribution<float>(3.5f);
boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> > Sampler::sGenerator = boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> >(sRNG, sDistribution);
	
Sampler::Sampler(const PointCloud &pointCloud, short minNumAngles)
	: mPointCloud(pointCloud)
	, mMinNumAngles(minNumAngles)
	, mNumEmptyAngles(mPointCloud.numAngles())
	, mNumAvailablePoints(numPoints())
	, mCurrentAngle(0)
{  
	mPoints = new size_t[numPoints()];
	mNumPointsPerAngle = (size_t*)calloc(mPointCloud.numAngles(), sizeof(size_t));
	mTotalNumPointsPerAngle = (size_t*)calloc(mPointCloud.numAngles(), sizeof(size_t));
	mNumPicksPerPoint = new size_t[numPoints()];
	for (size_t i = 0; i < numPoints(); i++)
	{
		mPoints[i] = i;
		mNumPicksPerPoint[i] = 0;
		++mTotalNumPointsPerAngle[angleIndex(i)];
		if (++mNumPointsPerAngle[angleIndex(i)] == 1)
			--mNumEmptyAngles;
	}
	mPointsPerAngle = new size_t*[mPointCloud.numAngles()];
	int *indices = (int*)calloc(mPointCloud.numAngles(), sizeof(int));
	for (short i = 0; i < mPointCloud.numAngles(); i++)
	{
		mPointsPerAngle[i] = new size_t[mTotalNumPointsPerAngle[i]];
	}
	for (size_t i = 0; i < numPoints(); i++)
	{
		short angle = angleIndex(i);
		mPointsPerAngle[angle][indices[angle]] = i;
		++indices[angle];
	}
	delete[] indices;
}

Sampler::~Sampler()
{
	delete[] mPoints;
	delete[] mNumPointsPerAngle;
	delete[] mTotalNumPointsPerAngle;
	delete[] mNumPicksPerPoint;
	for (short i = 0; i < mPointCloud.numAngles(); i++)
	{
		delete[] mPointsPerAngle[i];
	}
	delete[] mPointsPerAngle;
}

bool Sampler::canSample() const 
{
	return mPointCloud.numGroups() > mMinNumAngles && mNumAvailablePoints > 2 * mMinNumAngles && 
		(mPointCloud.numAngles() - mNumEmptyAngles) > mMinNumAngles;
}

std::pair<size_t, size_t> Sampler::sample()  
{
	std::pair<size_t, size_t> p;
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.first = selectRandomPoint();
	if (++mNumPicksPerPoint[p.first] > mPointCloud.point(p.first).maxNumSamples)
	{
		removePoint(p.first);
	}
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeSample1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.second = selectAnotherRandomPoint(p.first);
	if (++mNumPicksPerPoint[p.second] > mPointCloud.point(p.first).maxNumSamples)
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

void Sampler::addPoint(size_t point)
{
	if (!isRemoved(point)) return;
	mPoints[point] = point;
	if (mNumPointsPerAngle[angleIndex(point)]++ == 0)
		--mNumEmptyAngles;
	++mNumAvailablePoints;
	mNumPicksPerPoint[point] = 0;
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
	size_t point = (size_t)(rand() % n);
	return getPoint(mPointsPerAngle[angle][point]);
}

size_t Sampler::selectRandomPointWithValidAngle(short angle) 
{
	size_t point = selectRandomPointWithAngle(angle);
	if (point < numPoints()) return point;
	short clockwiseAngle = angle;
	short counterClockwiseAngle = angle;
	for (short i = 0; i < mPointCloud.numAngles() / 2; i++)
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
	//size_t point = (mLastPoint + (size_t)(numPoints() - getRandomValueFromExponentialDist() * numPoints() / 2)) % numPoints();
	//mLastPoint = getPoint(point);
	//return mLastPoint;
	//size_t point = (size_t)(rand() % numPoints());
	//return getPoint(point);
	size_t point = selectRandomPointWithValidAngle(mCurrentAngle);
	++mCurrentAngle;
	return point;
}

size_t Sampler::selectAnotherRandomPoint(size_t point)
{
	short angle = (short)(getRandomValueFromExponentialDist() * (mPointCloud.numAngles()) + angleIndex(point) + mPointCloud.numAngles() / 4) % mPointCloud.numAngles();
	return selectRandomPointWithValidAngle(angle);
}
	
