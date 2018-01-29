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

class PointCloud
{
public:
	static void createPointCloudsFromImage(const cv::Mat &img, int cannyLowThreshold, short numAngles, std::vector<PointCloud> &pointClouds);

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
		return mRect;
	}
	
private:
	std::vector<Point> mPoints;
	std::vector<Point> mGroups;
	cv::Rect2f mRect;
	short mNumAngles;
	Point p;
				
	PointCloud(int numAngles);
	
	void setExtension();
	
	void sortPointsByCurvature();
	
};

#endif // _POINTCLOUD_H_
