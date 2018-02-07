#include "pointcloud.h"

#include <iostream>
#include <map>

#include "sampler.h"

cv::Mat PointCloud::sEdgeImg;
std::vector<Point*> PointCloud::sPoints;

cv::Rect2f PointCloud::createPointCloudsFromImage(const cv::Mat &img, int cannyLowThreshold, short numAngles, short minNumAngles, std::vector<PointCloud*> &pointClouds)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	ImageUtils imgUtils(img, numAngles, cannyLowThreshold);
	sEdgeImg = imgUtils.edgeImg();
	for (Point *point : sPoints)
	{
		if (point != NULL)
			delete point;
	}
	sPoints = std::vector<Point*>(sEdgeImg.total(), NULL);
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
	pointClouds = std::vector<PointCloud*>(numConnectedComponents, NULL);
	for (size_t i = 0; i < numConnectedComponents; i++)
	{
		if (imgUtils.countPointsInComponent(i) > minNumAngles)
		{
			pointClouds[i] = new PointCloud(numAngles);
		}
	}
	for (int edgeIndex = 0; edgeIndex < imgUtils.numEdges(); edgeIndex++)
	{
		int label = imgUtils.labelOf(edgeIndex);
		if (pointClouds[label] == NULL) continue;
		Point *point = new Point;
		size_t reverseIndex = imgUtils.reverseIndexOf(edgeIndex);
		point->position = cv::Point2f(reverseIndex % img.cols, reverseIndex / img.cols);
		point->normal = imgUtils.normal(edgeIndex);
		point->angleIndex = imgUtils.angleIndexOf(edgeIndex);
		point->curvature = imgUtils.curvature(edgeIndex);
		pointClouds[label]->mPoints.push_back(point);
		pointClouds[label]->mAngles.insert(point->angleIndex);
		point->pointCloud = pointClouds[label];
		sPoints[reverseIndex] = point;
	}	
	#ifdef _BENCHMARK
		end = std::chrono::high_resolution_clock::now();
		gTimeCreatePointCloud += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif
	int anglesPerIndex = 360 / numAngles;
	for (int i = 0; i < numConnectedComponents; i++)
	{
		if (pointClouds[i] == NULL) continue;
		pointClouds[i]->setExtension();
		if (pointClouds[i]->mPoints.size() >= minNumAngles && pointClouds[i]->mAngles.size() >= minNumAngles)
		{
			std::sort(pointClouds[i]->mPoints.begin(), pointClouds[i]->mPoints.end(), [](const Point *a, const Point *b)
			{
				return (a->angleIndex < b->angleIndex) || ((a->angleIndex == b->angleIndex) && (a->curvature > b->curvature));
			});
			for (size_t j = 0; j < pointClouds[i]->mPoints.size(); j++)
			{
				pointClouds[i]->mPoints[j]->index = j;
			}
		}
	}
	cv::Rect extension = getExtension(pointClouds);

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

void PointCloud::setExtension()
{
	float minX, minY, maxX, maxY;
	minX = minY = std::numeric_limits<float>::max();
	maxX = maxY = -std::numeric_limits<float>::max();
	for (const Point *point : mPoints)
	{
		float x = point->position.x;
		float y = point->position.y;
		if (x < minX) minX = x;
		if (x > maxX) maxX = x;
		if (y < minY) minY = y;
		if (y > maxY) maxY = y;
	}
	mExtension = cv::Rect2f(minX, minY, maxX - minX, maxY - minY); 
}

cv::Rect2f PointCloud::getExtension(const std::vector<PointCloud*> &pointClouds)
{
	float minX, minY, maxX, maxY;
	minX = minY = std::numeric_limits<float>::max();
	maxX = maxY = -std::numeric_limits<float>::max();
	for (const PointCloud *pointCloud : pointClouds)
	{
		if (pointCloud == NULL) continue;
		float x1 = pointCloud->mExtension.x;
		float y1 = pointCloud->mExtension.y;
		float x2 = pointCloud->mExtension.x + pointCloud->mExtension.width;
		float y2 = pointCloud->mExtension.y + pointCloud->mExtension.height;
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

void PointCloud::createSampler(short minArcLength)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	mSampler = new Sampler(*this, minArcLength);
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeCreateSampler += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
}

