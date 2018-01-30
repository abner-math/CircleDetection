#ifndef _HOUGH_ACCUMULATOR_H_
#define _HOUGH_ACCUMULATOR_H_

#include <set>

#include "sampler.h"

struct Circle
{
	cv::Point2f center;
	float radius;
};

struct Intersection
{
	Sampler *sampler;
	size_t p1;
	size_t p2;
	cv::Point2f position;
	float dist;
};

class HoughCell;

class HoughAccumulator
{
public:
	HoughAccumulator(HoughCell *cell);
	
	~HoughAccumulator();
	
	HoughCell* cell()  
	{
		return mCell;
	}
	
	float radius() const;
	
	const std::vector<Intersection>& intersections() const 
	{
		return mIntersections;
	}
	
	void accumulate(const Intersection &intersection);
	
	bool hasCircleCandidate() const;
	
	Circle getCircleCandidate() const;
	
private:
	HoughCell *mCell;
	std::set<size_t> mAngles;
	std::set<float> mRadius;
	std::vector<Intersection> mIntersections;
	
};

#endif // _HOUGH_ACCUMULATOR_H_
