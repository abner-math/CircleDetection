#include "houghaccumulator.h"

#include <iostream>

#include "houghcell.h"

HoughAccumulator::HoughAccumulator(HoughCell *cell, float radius)
	: mCell(cell)
	, mRadius(radius)
	, mNumAngles(0)
	, mVisited(false)
{
	mAngles = (bool*)calloc(cell->numAngles(), sizeof(bool));
}

HoughAccumulator::~HoughAccumulator()
{
	delete[] mAngles;
}

void HoughAccumulator::accumulate(const Intersection &intersection)
{
	size_t angle1 = intersection.sampler->pointCloud().group(intersection.p1).angleIndex;
	size_t angle2 = intersection.sampler->pointCloud().group(intersection.p2).angleIndex;
	if (!mAngles[angle1])
	{
		mAngles[angle1] = true;
		++mNumAngles;
	}
	if (!mAngles[angle2])
	{
		mAngles[angle2] = true;
		++mNumAngles;
	}
	mIntersections.push_back(intersection);
}

bool HoughAccumulator::hasEllipseCandidate() const 
{
	return mIntersections.size() > 6 && mNumAngles >= mCell->minNumAngles();
}

Ellipse HoughAccumulator::getEllipseCandidate()  
{
	//mVisited = true;
	/*std::vector<float> xs, ys;
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
	// sort intersections by dist from center 
	std::vector<Intersection> newIntersections(mIntersections.begin(), mIntersections.end());
	for (Intersection &intersection : newIntersections)
	{
		intersection.dist = norm(intersection.position - center);
	}
	std::sort(newIntersections.begin(), newIntersections.end(), [](const Intersection &a, const Intersection &b)
	{
		return a.dist < b.dist;
	});*/
	// least squares with only first half of intersections
	/*Eigen::Matrix2Xf points(2, newIntersections.size() / 2 * 2);
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
	EllipseFunctor functor(points);
	Eigen::LevenbergMarquardt<EllipseFunctor, float> lm(functor);
	lm.minimize(params);
	Ellipse circle;
	circle.center = cv::Point2f(params(0), params(1));
	circle.radius = std::abs(params(2));*/
	std::vector<cv::Point2f> points;
	for (size_t i = 0; i < mIntersections.size(); i++)
	{
		cv::Point2f p1 = mIntersections[i].sampler->pointCloud().group(mIntersections[i].p1).position;
		cv::Point2f p2 = mIntersections[i].sampler->pointCloud().group(mIntersections[i].p2).position;
		points.push_back(p1);
		points.push_back(p2);
	}
	Ellipse ellipse;
	ellipse.ellipse = cv::fitEllipse(points);
	return ellipse;
}

