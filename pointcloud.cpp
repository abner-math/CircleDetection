#include "pointcloud.h"

#include <iostream>

double PointCloud::normalizeAngle(double angleDegrees)
{
	if (angleDegrees > 180) 
		angleDegrees -= 180;
	return angleDegrees;
}

std::vector<PointCloud*> PointCloud::createPointCloudsFromImage(const cv::Mat &img, int cannyLowThreshold, size_t numAngles)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	ImageUtils imgUtils(img, cannyLowThreshold);
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeProcessImage += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif 
	PointCloud *pointCloud = NULL;
	std::vector<PointCloud*> pointClouds;
	int index = 0;
	std::vector<PointCloud*> indices(img.cols);
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (imgUtils.edge(index))
			{
				
				if (pointCloud == NULL)
				{
					pointCloud = new PointCloud(numAngles);
					pointClouds.push_back(pointCloud);
				}
				Point point;
				point.position = cv::Point2f(x, y);
				point.normal = imgUtils.sobel(index);
				point.inverseNormal = imgUtils.inverseSobel(index);
				point.normalAngle = normalizeAngle(imgUtils.sobelAngle(index));
				point.normalAngleIndex = getNormalAngleIndex(point.normalAngle);
				point.curvature = imgUtils.curvature(index);
				pointCloud->addPoint(point);
				indices[x] = pointCloud;
			}
			else
			{
				indices[x] = NULL;
			}
			++index;
		}
	}
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeCreatePointCloud += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif
}

PointCloud::PointCloud(const cv::Mat &img, int cannyLowThreshold, size_t numAngles)
	: mNumAngles(numAngles)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	ImageUtils imgUtils(img, cannyLowThreshold);
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeProcessImage += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif 
	mPoints.reserve(img.total());
	int index = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (imgUtils.edge(index))
			{
				Point point;
				point.position = cv::Point2f(x, y);
				point.normal = imgUtils.sobel(index);
				point.inverseNormal = imgUtils.inverseSobel(index);
				point.normalAngle = normalizeAngle(imgUtils.sobelAngle(index));
				point.normalAngleIndex = getNormalAngleIndex(point.normalAngle);
				if (!std::isnan(point.curvature) && !std::isinf(point.curvature))
					mPoints.push_back(point);
			}
			++index;
		}
	}
	setExtension();
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeCreatePointCloud += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif
}

void PointCloud::setExtension()
{
	float minX, minY, maxX, maxY;
	minX = minY = std::numeric_limits<float>::max();
	maxX = maxY = -std::numeric_limits<float>::max();
	for (const Point &point : mPoints)
	{
		float x = point.position.x;
		float y = point.position.y;
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

