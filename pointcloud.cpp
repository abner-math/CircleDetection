#include "pointcloud.h"

#include <iostream>

std::vector<PointCloud*> PointCloud::createPointCloudsFromImage(const cv::Mat &img, int cannyLowThreshold, short numAngles)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	ImageUtils imgUtils(img, numAngles, cannyLowThreshold);
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
	std::cout << "Num groups: " << imgUtils.groupPointsByAngle() << " (previously " << imgUtils.numEdges() << ": " << (1 - imgUtils.groupPointsByAngle() / (float)imgUtils.numEdges()) * 100 << "% reduction)" << std::endl;
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeGroupPoints += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		begin = std::chrono::high_resolution_clock::now();
	#endif
	std::vector<PointCloud*> pointClouds(numConnectedComponents);
	for (int i = 0; i < numConnectedComponents; i++)
	{
		pointClouds[i] = new PointCloud(numAngles);
	}
	for (int edgeIndex = 0; edgeIndex < imgUtils.numEdges(); edgeIndex++)
	{
		Point point;
		point.position = imgUtils.position(edgeIndex);
		point.normal = imgUtils.sobel(edgeIndex);
		point.inverseNormal = imgUtils.inverseSobel(edgeIndex);
		point.angleIndex = imgUtils.angleIndexOf(edgeIndex);
		point.curvature = imgUtils.curvature(edgeIndex);
		point.isCentroid = imgUtils.isCentroid(edgeIndex);
		pointClouds[imgUtils.labelOf(edgeIndex)]->addPoint(point);
	}
	for (int i = 0; i < numConnectedComponents; i++)
	{
		pointClouds[i]->setExtension();
		pointClouds[i]->sortPointsByCurvature();
	}
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeCreatePointCloud += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif
	return pointClouds;
}

PointCloud::PointCloud(int numAngles)
	: mNumAngles(numAngles)
{
	
}

void PointCloud::addPoint(const Point &point)
{
	mPoints.push_back(point);
	if (point.isCentroid)
		mGroups.push_back(point);
}
	
void PointCloud::setExtension()
{
	float minX, minY, maxX, maxY;
	minX = minY = std::numeric_limits<float>::max();
	maxX = maxY = -std::numeric_limits<float>::max();
	for (const Point &point : mGroups)
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

void PointCloud::sortPointsByCurvature()
{
	std::sort(mGroups.begin(), mGroups.end(), [](const Point &a, const Point &b)
	{
		return a.curvature > b.curvature;
	});
}

