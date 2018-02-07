#include "houghaccumulator.h"

#include <iostream>

#include "houghcell.h"

HoughAccumulator::HoughAccumulator(HoughCell *cell, float radius)
	: mCell(cell)
	, mRadius(radius)
{
	
}

void HoughAccumulator::accumulate(const Intersection &intersection)
{
	const Point &p1 = intersection.sampler->pointCloud().group(intersection.p1);
	const Point &p2 = intersection.sampler->pointCloud().group(intersection.p2);
	short angle1 = p1.angleIndex;
	short angle2 = p2.angleIndex;
	mAngles.insert(angle1);
	mAngles.insert(angle2);
	mIntersections.push_back(intersection);
	mPositionsX.push_back(p1.position.x);
	mPositionsY.push_back(p1.position.y);
	mPositionsX.push_back(p2.position.x);
	mPositionsY.push_back(p2.position.y);
}

bool HoughAccumulator::hasCandidate() const 
{
	return mIntersections.size() > 6 && mAngles.size() >= mCell->minNumAngles();
}

cv::RotatedRect HoughAccumulator::ellipseEquationToRect(EllipseEquation &equation)
{
	double centerX, centerY, majorAxisLength, minorAxisLength;
	ComputeEllipseCenterAndAxisLengths(&equation, &centerX, &centerY, &majorAxisLength, &minorAxisLength);
	float rotationAngle = (float)std::atan(equation.B() / (equation.A() - equation.C())) / 2 * 180 / M_PI;
	cv::Point2f center((float)centerX, (float)centerY);
	cv::Size2f size((float)minorAxisLength * 2, (float)majorAxisLength * 2);
	return cv::RotatedRect(center, size, rotationAngle);
}
	
Ellipse HoughAccumulator::getEllipseCandidate()  
{
	Ellipse ellipse;
	ellipse.falsePositive = true;
	EllipseEquation equation;
	if (EllipseFit(mPositionsX.data(), mPositionsY.data(), mPositionsX.size(), &equation))//, BOOKSTEIN))
	{
		ellipse.falsePositive = false;
		ellipse.equation = equation;
		ellipse.rect = ellipseEquationToRect(equation);
	}
	return ellipse;
}

Circle HoughAccumulator::getCircleCandidate()
{
	Circle circle;
	circle.falsePositive = true;
	double centerX, centerY, radius, error;
	if (CircleFit(mPositionsX.data(), mPositionsY.data(), mPositionsX.size(), &centerX, &centerY, &radius, &error))
	{
		circle.falsePositive = false;
		circle.center = cv::Point2f((float)centerX, (float)centerY);
		circle.radius = (float)radius;
	}
	return circle;
}
