#include "houghaccumulator.h"

#include <iostream>

#include "houghcell.h"

HoughAccumulator::HoughAccumulator(HoughCell *cell)
	: mCell(cell)
{
	
}

HoughAccumulator::~HoughAccumulator()
{
	
}

float HoughAccumulator::radius() const 
{
	auto it = mRadius.begin();
	std::advance(it, mRadius.size() / 2);
	return *it;
}

void HoughAccumulator::accumulate(const Intersection &intersection)
{
	mRadius.insert(intersection.dist);
	mIntersections.push_back(intersection);
	mAngles.insert(mCell->pointCloud()->point(intersection.p1).angleIndex);
	mAngles.insert(mCell->pointCloud()->point(intersection.p2).angleIndex);
}

bool HoughAccumulator::hasCircleCandidate() const 
{
	return mAngles.size() >= mCell->minArcLength() && mIntersections.size() > 4;
}

Circle* HoughAccumulator::getCircleCandidate() const 
{
	Circle *circle = new Circle;
	std::vector<float> xs, ys;
	for (const Intersection &intersection : mIntersections)
	{
		xs.push_back(intersection.position.x);
		ys.push_back(intersection.position.y);
	}
	size_t median = mIntersections.size() / 2;
	// center = median of positions 
	std::nth_element(xs.begin(), xs.begin() + median, xs.end());
	std::nth_element(ys.begin(), ys.begin() + median, ys.end());
	circle->center = cv::Point2f(xs[median], ys[median]);
	circle->radius = radius();
	// least squares 
	/*int col = 0;
	Eigen::Matrix2Xf points(2, mIntersections.size() * 2);
	for (const std::shared_ptr<const Intersection> &intersection : mIntersections)
	{
		points.col(col++) = Eigen::Vector2f(intersection->p1->position.x, intersection->p1->position.y);
		points.col(col++) = Eigen::Vector2f(intersection->p2->position.x, intersection->p2->position.y);
	}
	Eigen::VectorXf params(3);
	params << circle->center.x, circle->center.y, circle->radius;
	CircleFunctor functor(points);
	Eigen::LevenbergMarquardt<CircleFunctor, float> lm(functor);
	lm.minimize(params);
	circle->center = cv::Point2f(params(0), params(1));
	circle->radius = std::abs(params(2));*/
	return circle;
}

