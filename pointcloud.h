#ifndef _POINTCLOUD_H_
#define _POINTCLOUD_H_

#include "imageutils.h"
#include "benchmark.hpp"
 
struct Point
{
	cv::Point2f position; 
	cv::Point2f normal;
	cv::Point2f inverseNormal;
	double normalAngle;
	size_t normalAngleIndex;
	float curvature;
	
	bool operator<(const Point &otherPoint) const 
	{
		return curvature > otherPoint.curvature;
	}
};

inline float norm(const cv::Point2f &p)
{
	return std::sqrt(p.x * p.x + p.y * p.y);
}

class PointCloud
{
public:
	PointCloud(const cv::Mat &img, int cannyLowThreshold, size_t numAngles);

	size_t numAngles() const 
	{
		return mNumAngles;
	}
	
	size_t numPoints() const 
	{
		return mPoints.size();
	}
	
	const Point& point(size_t index) const 
	{
		return mPoints[index];
	}
	
	const cv::Rect2f& extension() const 
	{
		return mRect;
	}
	
private:
	std::vector<Point> mPoints;
	cv::Rect2f mRect;
	size_t mNumAngles;
				
	inline size_t getNormalAngleIndex(double normalAngle)
	{
		return static_cast<size_t>(std::round(normalAngle / (180.0 / mNumAngles))) % mNumAngles;
	}
	
	double normalizeAngle(double angleDegrees);

	void setExtension();
	
};

#endif // _POINTCLOUD_H_
