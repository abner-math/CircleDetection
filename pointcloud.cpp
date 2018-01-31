#include "pointcloud.h"

#include <iostream>

#include "sampler.h"

cv::Rect2f PointCloud::createPointCloudsFromImage(const cv::Mat &img, int cannyLowThreshold, short numAngles, std::vector<PointCloud> &pointClouds)
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
	pointClouds = std::vector<PointCloud>(numConnectedComponents, PointCloud(numAngles));
	for (int i = 0; i < numConnectedComponents; i++)
	{
		pointClouds[i].setCenter(imgUtils.center(i));
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
		pointClouds[imgUtils.labelOf(edgeIndex)].addPoint(point);
	}
	for (int i = 0; i < numConnectedComponents; i++)
	{
		pointClouds[i].setExtension();
		pointClouds[i].sortPointsByCurvature();
	}
	std::sort(pointClouds.begin(), pointClouds.end(), [](const PointCloud &a, const PointCloud &b)
	{
		return a.group(a.numGroups() / 2).curvature > b.group(b.numGroups() / 2).curvature;
	});
	cv::Rect extension = getExtension(pointClouds);
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeCreatePointCloud += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif
	return extension;
}

PointCloud::PointCloud(int numAngles)
	: mNumAngles(numAngles)
	, mSampler(NULL)
{
	
}

PointCloud::~PointCloud()
{
	if (mSampler != NULL)
		delete mSampler;
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
	//float centerX = (minX + maxX) / 2;
	//float centerY = (minY + maxY) / 2;
	//minX = centerX - size / 2;
	//minY = centerY - size / 2;
	mExtension = cv::Rect2f(minX, minY, size, size); 
}

cv::Rect2f PointCloud::getExtension(const std::vector<PointCloud> &pointClouds)
{
	float minX, minY, maxX, maxY;
	minX = minY = std::numeric_limits<float>::max();
	maxX = maxY = -std::numeric_limits<float>::max();
	for (const PointCloud &pointCloud : pointClouds)
	{
		float x1 = pointCloud.mExtension.x;
		float y1 = pointCloud.mExtension.y;
		float x2 = pointCloud.mExtension.x + pointCloud.mExtension.width;
		float y2 = pointCloud.mExtension.y + pointCloud.mExtension.height;
		if (x1 < minX) minX = x1;
		if (y1 < minY) minY = y1;
		if (x2 > maxX) maxX = x2;
		if (y2 > maxY) maxY = y2;
	}
	float size = std::max(maxX - minX, maxY - minY) / 2;
	float centerX = (minX + maxX) / 2;
	float centerY = (minY + maxY) / 2;
	minX = centerX - size;
	minY = centerY - size;
	return cv::Rect2f(minX, minY, size*2, size*2); 
}

void PointCloud::sortPointsByCurvature()
{
	std::sort(mGroups.begin(), mGroups.end(), [](const Point &a, const Point &b)
	{
		return a.curvature > b.curvature;
	});
}

void PointCloud::createSampler(short minArcLength, float minQuadtreeSize)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	mSampler = new Sampler(*this, minQuadtreeSize, minArcLength);
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeCreateSampler += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
}
