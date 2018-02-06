#include "houghaccumulator.h"

#include <iostream>

#include "houghcell.h"
#include "EllipseFit.h"

using namespace cv;
using namespace std;
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
	mVisited = true;
	/*std::vector<cv::Point2f> points;
	for (size_t i = 0; i < mIntersections.size(); i++)
	{
		cv::Point2f p1 = mIntersections[i].sampler->pointCloud().group(mIntersections[i].p1).position;
		cv::Point2f p2 = mIntersections[i].sampler->pointCloud().group(mIntersections[i].p2).position;
		points.push_back(p1);
		points.push_back(p2);
	}
	Ellipse ellipse;
	ellipse.ellipse = cv::fitEllipse(points);*/
	double *xs = new double[mIntersections.size() * 2];
	double *ys = new double[mIntersections.size() * 2];
	for (size_t i = 0; i < mIntersections.size(); i++)
	{
		cv::Point2f p1 = mIntersections[i].sampler->pointCloud().group(mIntersections[i].p1).position;
		cv::Point2f p2 = mIntersections[i].sampler->pointCloud().group(mIntersections[i].p2).position;
		xs[i * 2] = p1.x;
		xs[i * 2 + 1] = p2.x;
		ys[i * 2] = p1.y;
		ys[i * 2 + 1] = p2.y; 
	}
	Ellipse ellipse;
	ellipse.falsePositive = true;
	EllipseEquation equation;
	if (EllipseFit(xs, ys, mIntersections.size() * 2, &equation, BOOKSTEIN))
	{
		ellipse.falsePositive = false;
		double centerX, centerY, majorAxisLength, minorAxisLength;
		ComputeEllipseCenterAndAxisLengths(&equation, &centerX, &centerY, &majorAxisLength, &minorAxisLength);
		float rotationAngle = (float)std::atan(equation.B() / (equation.A() - equation.C())) / 2 * 180 / M_PI;
		cv::Point2f center((float)centerX, (float)centerY);
		cv::Size2f size((float)minorAxisLength * 2, (float)majorAxisLength * 2);
		ellipse.ellipse = cv::RotatedRect(center, size, rotationAngle);
	}
	delete[] xs;
	delete[] ys;
	return ellipse;
}

