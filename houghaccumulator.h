#ifndef _HOUGH_ACCUMULATOR_H_
#define _HOUGH_ACCUMULATOR_H_

#include <set>

#include "sampler.h"
#include "circlefunctor.h"

struct Circle
{
	cv::Point2f center;
	float radius;
	bool removed;
};

struct Intersection
{
	Sampler *sampler;
	size_t p1;
	size_t p2;
	cv::Point2f position;
	float dist;
	
	bool operator<(const Intersection &other) const 
	{
		return dist < other.dist;
	}
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
	
	const std::set<Intersection>& intersections() const 
	{
		return mIntersections;
	}
	
	void accumulate(const Intersection &intersection);
	
	bool hasCircleCandidate() const;
	
	Circle getCircleCandidate() const;
	
private:
	HoughCell *mCell;
	std::set<size_t> mAngles;
	std::set<Intersection> mIntersections;
	
};

#endif // _HOUGH_ACCUMULATOR_H_
