#ifndef _POINTCLOUD_H_
#define _POINTCLOUD_H_

#include "imageutils.h"
#include "benchmark.hpp"
 
struct Point
{
	cv::Point2f position; 
	cv::Point2f normal;
	cv::Point2f inverseNormal;
	int angleIndex;
	float curvature;
	bool isCentroid;
};

inline float norm(const cv::Point2f &p)
{
	return std::sqrt(p.x * p.x + p.y * p.y);
}

class Sampler;

class PointCloud
{
public:
	static cv::Rect2f createPointCloudsFromImage(const cv::Mat &img, int cannyLowThreshold, short numAngles, std::vector<PointCloud> &pointClouds);

	~PointCloud();
	
	short numAngles() const 
	{
		return mNumAngles;
	}
	
	size_t numPoints() const 
	{
		return mPoints.size();
	}
	
	size_t numGroups() const 
	{
		return mGroups.size();
	}
	
	void addPoint(const Point &point);
	
	const Point& point(size_t index) const 
	{
		return mPoints[index];
	}
	
	const Point& group(size_t index) const 
	{
		return mGroups[index];
	}
	
	const cv::Rect2f& extension() const 
	{
		return mExtension;
	}
	
	void createSampler(short minArcLength, float minQuadtreeSize = 20.0f);
	
	Sampler* sampler() const 
	{
		return mSampler;
	}
	
private:
	std::vector<Point> mPoints;
	std::vector<Point> mGroups;
	short mNumAngles;
	cv::Rect2f mExtension;
	Sampler *mSampler;
				
	PointCloud(int numAngles);
	
	void sortPointsByCurvature();
	
	void setExtension();
	
	static cv::Rect2f getExtension(const std::vector<PointCloud> &pointClouds);
	
};

#endif // _POINTCLOUD_H_
