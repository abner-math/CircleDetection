#include "sampler.h"

#include <iostream>

boost::mt19937 Sampler::sRNG;
boost::exponential_distribution<float> Sampler::sDistribution = boost::exponential_distribution<float>(3.5f);
boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> > Sampler::sGenerator = boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> >(sRNG, sDistribution);
	
Sampler::Sampler(const PointCloud &pointCloud, short minNumAngles)
	: mPointCloud(pointCloud)
	, mMinNumAngles(minNumAngles)
	, mNumAvailablePoints(numPoints())
	, mNumEmptyAngles(0)
	, mCurrentAngle(0)
{  
	mPoints = new size_t[numPoints()];
	mCountPointsPerAngle = (size_t*)calloc(numAngles(), sizeof(size_t));
	mStartingIndicesPerAngle = new size_t[numAngles()];
	for (size_t i = 0; i < numPoints(); i++)
	{
		mPoints[i] = i;
		++mCountPointsPerAngle[angleIndex(i)];
	}
	size_t sum = 0;
	for (short angle = 0; angle < numAngles(); angle++)
	{
		mStartingIndicesPerAngle[angle] = sum;
		sum += mCountPointsPerAngle[angle];
	}
	for (short angle = 0; angle < numAngles(); angle++)
	{
		if (mCountPointsPerAngle[angle] == 0)
		{
			++mNumEmptyAngles;
		}
	}
}

Sampler::~Sampler()
{
	delete[] mPoints;
	delete[] mCountPointsPerAngle;
	delete[] mStartingIndicesPerAngle;
}

bool Sampler::canSample() const 
{
	return numPoints() > mMinNumAngles && (numAngles() - mNumEmptyAngles) > mMinNumAngles;
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
	p.second = selectAnotherRandomPoint(p.first);
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeSample2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return p;
}
	
void Sampler::removePoint(size_t point)
{
	if (isRemoved(point) || mNumAvailablePoints <= 3) return;
	--mNumAvailablePoints;
	short angle = angleIndex(point);
	size_t counterClockwiseIndex = mStartingIndicesPerAngle[angle] + mCountPointsPerAngle[angle] / 2;
	size_t clockwiseIndex = counterClockwiseIndex;
	for (size_t i = 0; i < numPoints() / 2; i++)
	{
		size_t index1 = mPoints[clockwiseIndex];
		if (!isRemoved(index1) && index1 != point)
		{
			mPoints[point] = index1;
			return;
		}
		clockwiseIndex = clockwiseIndex == numPoints() - 1 ? 0 : clockwiseIndex + 1;
		size_t index2 = mPoints[counterClockwiseIndex];
		if (!isRemoved(index2) && index2 != point)
		{
			mPoints[point] = index2;
			return;
		}
		counterClockwiseIndex = counterClockwiseIndex == 0 ? numPoints() - 1 : counterClockwiseIndex - 1;
	}
}

size_t Sampler::getValidPoint(size_t point)  
{
	if (point == mPoints[point]) return point;
	mPoints[point] = getValidPoint(mPoints[point]);
	return mPoints[point];
}

size_t Sampler::selectRandomPointFromAngle(short angle) 
{
	size_t count = mCountPointsPerAngle[angle];
	if (count == 0) return std::numeric_limits<size_t>::max();
	size_t index = rand() % count;
	size_t point = mPoints[mStartingIndicesPerAngle[angle] + index];
	return getValidPoint(point);
}

size_t Sampler::selectFromStartingAngle(short angle) 
{
	size_t point = selectRandomPointFromAngle(angle);
	if (point < numPoints()) return point;
	short clockwiseAngle = angle;
	short counterClockwiseAngle = angle;
	for (short i = 0; i < numAngles() / 2; i++)
	{
		clockwiseAngle = increaseOneAngle(clockwiseAngle);
		point = selectRandomPointFromAngle(clockwiseAngle);
		if (point < numPoints()) return point;
		counterClockwiseAngle = decreaseOneAngle(counterClockwiseAngle);
		point = selectRandomPointFromAngle(counterClockwiseAngle);
		if (point < numPoints()) return point;
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::selectRandomPoint()  
{
	size_t point = selectFromStartingAngle(mCurrentAngle);
	mCurrentAngle = (mCurrentAngle + 1) % numAngles();
	return point;
}

size_t Sampler::selectAnotherRandomPoint(size_t point)
{
	short randomAngle = (short)(getRandomValueFromExponentialDist() * numAngles());
	short orthogonalAngle = (randomAngle + angleIndex(point) + numAngles() / 4) % numAngles();
	return selectFromStartingAngle(orthogonalAngle);
}
	
