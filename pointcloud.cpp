#include "pointcloud.h"

#include <iostream>

PointCloud::PointCloud(const cv::Mat &gray, const cv::Mat &edges, size_t numAngles)
	: mNumAngles(numAngles)
{
	const uchar *edgePtr = (uchar*)edges.data;
	std::vector<Point*> points(edges.rows * edges.cols, NULL);
	ImageUtils imgUtils(gray);
	for (int y = 0, index = 0; y < gray.rows; y++)
	{
		for (int x = 0; x < gray.cols; x++, index++)
		{
			if (edgePtr[index] && x > 0 && y > 0 && x < gray.cols - 1 && y < gray.rows - 1)
			{
				int gradX = imgUtils.sobelX(x, y);
				int gradY = imgUtils.sobelY(x, y);
				float norm = std::sqrt(gradX * gradX + gradY * gradY);
				if (norm > std::numeric_limits<float>::epsilon())
				{
					Point *point = new Point;
					point->position = cv::Point2f(x + 0.5f, y + 0.5f);
					point->normal = cv::Point2f(gradX / norm, gradY / norm);
					point->inverseNormal = cv::Point2f(norm / gradX, norm / gradY);
					point->normalAngle = getNormalAngle(point->normal);
					point->normalAngleIndex = getNormalAngleIndex(point->normalAngle);
					points[index] = point;
					mPoints.push_back(point);
				}
			}
		}
	}
	for (int y = 0, index = 0; y < gray.rows; y++)
	{
		for (int x = 0; x < gray.cols; x++, index++)
		{
			if (points[index] != NULL)
			{
				points[index]->curvature = calculateCurvature(points, x, y, gray.cols);
			}
		}
	}
	sortPointsByCurvature();
	setExtension();
}

PointCloud::~PointCloud()
{
	for (auto it = mPoints.begin(); it != mPoints.end(); ++it)
	{
		delete *it;
	}
}

double PointCloud::getNormalAngle(const cv::Point2f &normal)
{
	double angleRadians = std::atan2(normal.y, normal.x) + M_PI;
	double angleDegrees = std::max(0.0, angleRadians * 180 / M_PI);
	if (angleDegrees > 180) 
		angleDegrees -= 180;
	return angleDegrees;
}

void PointCloud::setExtension()
{
	float minX, minY, maxX, maxY;
	minX = minY = std::numeric_limits<float>::max();
	maxX = maxY = -std::numeric_limits<float>::max();
	for (Point *point : mPoints)
	{
		float x = point->position.x;
		float y = point->position.y;
		if (x < minX) minX = x;
		if (x > maxX) maxX = x;
		if (y < minY) minY = y;
		if (y > maxY) maxY = y;
	}
	float size = std::max(maxX - minX, maxY - minY);
	float centerX = (minX + maxX) / 2;
	float centerY = (minY + maxY) / 2;
	minX = centerX - size;
	minY = centerY - size;
	mRect = cv::Rect2f(minX, minY, 2*size, 2*size); 
}

void PointCloud::sortPointsByCurvature()
{
	std::sort(mPoints.begin(), mPoints.end(), [](const Point* p1, const Point *p2)
	{
		return p1->curvature > p2->curvature;
	});
}

double PointCloud::calculateCurvature(const std::vector<Point*> &points, int x, int y, int cols)
{
	int centerIndex = y * cols + x;
	double sum = 0;
	int count = 0;
	for (int x_ = -2; x_ <= 2; x_++)
	{
		for (int y_ = -2; y_ <= 2; y_++)
		{
			int index = (y + y_) * cols + x + x_; 
			if (points[index] != NULL)
			{ 
				sum += std::acos(points[centerIndex]->normal.dot(points[index]->normal));
				++count;
			}
		}
	}
	return sum / count;
}

