#ifndef _HOUGH_ACCUMULATOR_H_
#define _HOUGH_ACCUMULATOR_H_

#include <set>

#include "sampler.h"
#include "circlefunctor.h"
#include "EllipseFit.h"
#include "CircleFit.h"

struct Ellipse
{
	cv::RotatedRect rect;
	EllipseEquation equation;
	bool falsePositive;
	float confidence;
};

struct Circle
{
	cv::Point2f center;
	float radius;
	bool falsePositive;
	float confidence;
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
	HoughAccumulator(HoughCell *cell, float radius);
	
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
	
	bool isVisited() const 
	{
		return mVisited;
	}
	
	void setVisited()
	{
		mVisited = true;
	}
	
	void accumulate(const Intersection &intersection);
	
	bool hasCandidate() const;
	
	Ellipse getEllipseCandidate();
	
	Circle getCircleCandidate();
	
private:
	HoughCell *mCell;
	float mRadius;
	bool mVisited;
	std::set<short> mAngles;
	std::vector<Intersection> mIntersections;
	std::vector<double> mPositionsX;
	std::vector<double> mPositionsY;
	
	cv::RotatedRect ellipseEquationToRect(EllipseEquation &equation);
	
};

#endif // _HOUGH_ACCUMULATOR_H_
