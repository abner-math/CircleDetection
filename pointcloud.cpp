#include "pointcloud.h"

#include <iostream>

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
	int numConnectedComponents = imgUtils.createConnectedComponents();
	std::cout << "Num connected components: " << numConnectedComponents << std::endl;
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeCreateConnectedComponents += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif
	std::vector<PointCloud*> pointClouds(numConnectedComponents);
	for (int i = 0; i < numConnectedComponents; i++)
	{
		pointClouds[i] = new PointCloud;
	}
	int index = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (imgUtils.isEdge(index))
			{
				Point point;
				point.position = cv::Point2f(x, y);
				point.normal = imgUtils.sobel(index);
				point.inverseNormal = imgUtils.inverseSobel(index);
				point.normalAngle = imgUtils.sobelAngle(index);
				point.normalAngleIndex = getNormalAngleIndex(point.normalAngle, numAngles);
				point.curvature = imgUtils.curvature(index);
				pointClouds[imgUtils.labelOf(index)]->addPoint(point);
			}
			++index;
		}
	}
	for (int i = 0; i < numConnectedComponents; i++)
	{
		pointClouds[i]->setExtension();
	}
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeCreatePointCloud += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif
	return pointClouds;
}

void PointCloud::addPoint(const Point &point)
{
	mPoints.push_back(point);
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

