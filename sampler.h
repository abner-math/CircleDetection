#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include <random>
#include <set>
#include <map>

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

#include "pointcloud.h"

#define MAX_NUM_PICKS 5

class Sampler
{
public:
	Sampler(const PointCloud &pointCloud, short minNumAngles);
	
	~Sampler();
	
	size_t numAvailablePoints() const 
	{
		return mNumAvailablePoints;
	}
	
	size_t numPoints() const 
	{
		return mPointCloud.numPoints();
	}
	
	short numAngles() const 
	{
		return mPointCloud.numAngles();
	}
	
	short minNumAngles() const 
	{
		return mMinNumAngles;
	}
	
	const PointCloud& pointCloud() const 
	{
		return mPointCloud;
	}
	
	bool canSample() const;
	
	std::pair<size_t, size_t> sample();
	
	void removePoint(size_t point);
	
	bool isRemoved(size_t point) const
	{
		if (mPoints == NULL) return true;
		return mPoints[point] != point;
	}
	
private:
	static boost::mt19937 sRNG;
	static boost::exponential_distribution<float> sDistribution;
	static boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> > sGenerator;
	const PointCloud &mPointCloud;
	short mMinNumAngles;
	size_t *mPoints;
	std::map<short, size_t> mCountPointsPerAngle;
	std::map<size_t, size_t> mStartingIndicesPerAngle;
	std::set<short>::const_iterator mCurrentAngle;
	std::map<short, short> mOppositeAngles;
	size_t *mNumPicks;
	size_t mNumAvailablePoints;
	
	short angleIndex(size_t point) const 
	{
		return mPointCloud.point(point).angleIndex;
	}
	
	size_t getValidPoint(size_t index);
	
	size_t selectRandomPointFromAngle(short angle);
	 
	size_t selectRandomPoint();
	
	size_t selectAnotherRandomPoint(size_t point);
	
	static float getRandomValueFromExponentialDist()  
	{
		float value;
		do
		{
			value = sGenerator();
		} while (value >= 1.0f);
		return value;
	}
	
};

#endif // _SAMPLER_H_
