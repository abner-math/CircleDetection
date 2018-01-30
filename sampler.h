#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include <random>
#include <set>

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

#include "pointcloud.h"

class Sampler
{
public:
	Sampler(const PointCloud &pointCloud, float minQuadtreeSize, short minArcLength);
	
	~Sampler();
	
	size_t numAvailablePoints() const 
	{
		return mNumAvailablePoints;
	}
	
	size_t numPoints() const 
	{
		return mPointCloud.numGroups();
	}
	
	short numAngles() const 
	{
		return mPointCloud.numAngles();
	}
	
	const PointCloud& pointCloud() const 
	{
		return mPointCloud;
	}
	
	short minArcLength() const 
	{
		return mMinArcLength;
	}
	
	bool canSample() const;
	
	std::pair<size_t, size_t> sample();
	
	void removePoint(size_t point);
	
	bool isRemoved(size_t point) const
	{
		return mPoints[point] != point;
	}
	
private:
	static boost::mt19937 sRNG;
	static boost::exponential_distribution<float> sDistribution;
	static boost::variate_generator<boost::mt19937, boost::exponential_distribution<float> > sGenerator;
	const PointCloud &mPointCloud;
	const short mMinArcLength;
	size_t *mPoints;
	size_t **mPointsPerAngle;
	size_t *mTotalNumPointsPerAngle;
	size_t *mNumPointsPerAngle;
	size_t mNumEmptyAngles;
	size_t mNumAvailablePoints;
	
	short angleIndex(size_t point) const 
	{
		return mPointCloud.group(point).angleIndex;
	}
	
	size_t getPoint(size_t index);
	
	short decreaseOneAngle(short angle) const
	{
		if (angle == 0) return numAngles() - 1;
		return angle - 1;
	}
	
	short increaseOneAngle(short angle) const 
	{
		if (angle == numAngles() - 1) return 0;
		return angle + 1;
	}
	
	size_t selectRandomPointWithAngle(short angle);
	
	size_t selectRandomPointWithValidAngle(short angle);
	
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
