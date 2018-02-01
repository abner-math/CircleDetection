#ifndef _HOUGH_ACCUMULATOR_H_
#define _HOUGH_ACCUMULATOR_H_

#include <set>

#include "sampler.h"
#include "circlefunctor.h"

struct Ellipse
{
	cv::RotatedRect ellipse;
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
	HoughAccumulator(HoughCell *cell, float radius);
	
	~HoughAccumulator();
	
	HoughCell* cell()  
	{
		return mCell;
	}
	
	float radius() const
	{
		return mRadius;
	}
	
	const std::vector<Intersection>& intersections() const 
	{
		return mIntersections;
	}
	
	void accumulate(const Intersection &intersection);
	
	bool hasEllipseCandidate() const;
	
	Ellipse getEllipseCandidate();
	
	bool isVisited() const 
	{
		return mVisited;
	}
	
private:
	HoughCell *mCell;
	float mRadius;
	bool *mAngles;
	size_t mNumAngles;
	bool mVisited;
	std::vector<Intersection> mIntersections;
	
};

#endif // _HOUGH_ACCUMULATOR_H_
