#include "sampler.h"

#include <iostream>

boost::mt19937 Sampler::sRNG;
boost::exponential_distribution<float> Sampler::sDistribution = boost::exponential_distribution<float>(3.5f);
boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> > Sampler::sGenerator = boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> >(sRNG, sDistribution);
	
Sampler::Sampler(const PointCloud &pointCloud, short minNumAngles)
	: mPointCloud(pointCloud)
	, mPoints(NULL)
	, mMinNumAngles(minNumAngles)
	, mNumAvailablePoints(0) 
{  
	if (numPoints() >= mMinNumAngles && mPointCloud.angles().size() >= mMinNumAngles)
	{
		mPoints = new size_t[numPoints()];
		std::iota(mPoints, mPoints + numPoints(), 0);
		int lastAngle = angleIndex(0);
		mStartingIndicesPerAngle[lastAngle] = 0;
		mCurrentIndexInAngle[lastAngle] = 0;
		size_t count = 0;
		for (size_t i = 0; i < numPoints(); i++)
		{
			short angle = angleIndex(i);
			if (angle != lastAngle)
			{
				mStartingIndicesPerAngle[angle] = i;
				mCurrentIndexInAngle[angle] = 0;
				mCountPointsPerAngle[lastAngle] = count;
				count = 1;
				lastAngle = angle;
			}
			else
			{
				++count;
			}
		}
		mCountPointsPerAngle[lastAngle] = count;
		mCurrentAngle = mPointCloud.angles().begin();
		mNumAvailablePoints = numPoints();
		mNumPicks = (size_t*)calloc(numPoints(), sizeof(size_t));
		for (const short &angle1 : mPointCloud.angles())
		{
			short maxDist = 0;
			short maxAngle = angle1;
			for (const short &angle2 : mPointCloud.angles())
			{
				short diff = std::abs(angle1 - angle2);
				short dist = std::min(diff, (short)(mPointCloud.numAngles() - diff));
				if (dist > maxDist)
				{
					maxDist = dist;
					maxAngle = angle2;
				}
			}
			mOppositeAngles[angle1] = maxAngle;
		}
	}
}

Sampler::~Sampler()
{
	if (mPoints != NULL)
	{
		delete[] mPoints;
		delete[] mNumPicks;
	}
}

bool Sampler::canSample() const 
{
	return numAvailablePoints() >= mMinNumAngles && mPointCloud.angles().size() >= mMinNumAngles;
}

std::pair<size_t, size_t> Sampler::sample()  
{
	std::pair<size_t, size_t> p;
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.first = selectRandomPoint();
	++mNumPicks[p.first];
	if (mNumPicks[p.first] >= MAX_NUM_PICKS)
		removePoint(p.first);
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeSample1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif 
	p.second = selectAnotherRandomPoint(p.first);
	++mNumPicks[p.second];
	if (mNumPicks[p.second] >= MAX_NUM_PICKS)
		removePoint(p.second);
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeSample2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return p;
}
	
size_t Sampler::getValidPoint(size_t point)  
{
	if (point == mPoints[point]) return point;
	mPoints[point] = getValidPoint(mPoints[point]);
	return mPoints[point];
}

void Sampler::removePoint(size_t point)
{
	if (mPoints == NULL || isRemoved(point) || mNumAvailablePoints <= 1) return;
	--mNumAvailablePoints;
	size_t candidate1 = getValidPoint((point + 1) % numPoints());
	size_t candidate2 = getValidPoint(point == 0 ? numPoints() - 1 : point - 1);
	if (std::abs(candidate1 - point) < std::abs(candidate2 - point))
	{
		mPoints[point] = candidate1;
	}
	else
	{
		mPoints[point] = candidate2;
	}
}

size_t Sampler::selectRandomPointFromAngle(short angle) 
{
	size_t index = rand() % mCountPointsPerAngle[angle];
	return getValidPoint(mPoints[mStartingIndicesPerAngle[angle] + index]);
}

size_t Sampler::selectRandomPoint()  
{
	size_t point = getValidPoint(mPoints[mStartingIndicesPerAngle[*mCurrentAngle] + mCurrentIndexInAngle[*mCurrentAngle]]);
	mCurrentIndexInAngle[*mCurrentAngle] = (mCurrentIndexInAngle[*mCurrentAngle] + 1) % mCountPointsPerAngle[*mCurrentAngle];
	//size_t point = selectRandomPointFromAngle(*mCurrentAngle);
	if (++mCurrentAngle == mPointCloud.angles().end())
		mCurrentAngle = mPointCloud.angles().begin();
	return point;
}

size_t Sampler::selectAnotherRandomPoint(size_t point)
{
	return mPoints[mStartingIndicesPerAngle[mOppositeAngles[angleIndex(point)]]];
	//short angle = std::find(mAngles.begin(), mAngles.end(), angleIndex(point)) - mAngles.begin();
	//short randomAngle = (short)(getRandomValueFromExponentialDist() * mAngles.size());
	//short orthogonalAngle = mAngles[(randomAngle + angle + mAngles.size() / 2) % mAngles.size()];
	//return selectRandomPointFromAngle(orthogonalAngle);
}
	
