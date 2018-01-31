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
	auto it = mIntersections.begin();
	std::advance(it, mIntersections.size() / 2);
	return it->dist;
}

void HoughAccumulator::accumulate(const Intersection &intersection)
{
	mIntersections.insert(intersection);
	mAngles.insert(intersection.sampler->pointCloud().group(intersection.p1).angleIndex);
	mAngles.insert(intersection.sampler->pointCloud().group(intersection.p2).angleIndex);
}

bool HoughAccumulator::hasCircleCandidate() const 
{
	return mAngles.size() >= mCell->minArcLength() && mIntersections.size() > mCell->minArcLength();
}

Circle HoughAccumulator::getCircleCandidate() const 
{
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
	cv::Point2f center(xs[median], ys[median]);
	// radius = median of dists 
	std::vector<float> r;
	for (const Intersection &intersection : mIntersections)
	{
		r.push_back(norm(intersection.sampler->pointCloud().group(intersection.p1).position - center));
		r.push_back(norm(intersection.sampler->pointCloud().group(intersection.p2).position - center));
	}
	std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end());
	float newRadius = r[r.size() / 2];
	// sort intersections by dist from center 
	std::vector<Intersection> newIntersections(mIntersections.begin(), mIntersections.end());
	for (Intersection &intersection : newIntersections)
	{
		intersection.dist = norm(intersection.position - center);
	}
	std::sort(newIntersections.begin(), newIntersections.end(), [](const Intersection &a, const Intersection &b)
	{
		return a.dist < b.dist;
	});
	// least squares with only first half of intersections
	Eigen::Matrix2Xf points(2, newIntersections.size() / 2 * 2);
	int col = 0;
	for (int i = 0; i < newIntersections.size() / 2; i++)
	{
		cv::Point2f p1 = newIntersections[i].sampler->pointCloud().group(newIntersections[i].p1).position;
		cv::Point2f p2 = newIntersections[i].sampler->pointCloud().group(newIntersections[i].p2).position;
		points.col(col++) = Eigen::Vector2f(p1.x, p1.y);
		points.col(col++) = Eigen::Vector2f(p2.x, p2.y);
	}
	Eigen::VectorXf params(3);
	params << center.x, center.y, newRadius;
	CircleFunctor functor(points);
	Eigen::LevenbergMarquardt<CircleFunctor, float> lm(functor);
	lm.minimize(params);
	Circle circle;
	circle.center = cv::Point2f(params(0), params(1));
	circle.radius = std::abs(params(2));
	return circle;
}

