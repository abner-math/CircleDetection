#ifndef _POINTCLOUD_H_
#define _POINTCLOUD_H_

#include <set>

#include "imageutils.h"
#include "benchmark.hpp"
 
class PointCloud;

struct Point
{
	size_t index;
	cv::Point2f position; 
	cv::Point2f normal;
	int angleIndex;
	bool isGroup;
	PointCloud *pointCloud;
};

class Sampler; 

class PointCloud
{
public:
	static cv::Rect2f createPointCloudsFromImage(const cv::Mat &img, int cannyLowThreshold, short numAngles, short minNumAngles, std::vector<PointCloud> &pointClouds);

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
	
	const cv::Point2f& center() const 
	{
		return mCenter;
	}
	
	const Point& point(size_t index) const 
	{
		return *mPoints[index];
	}
	
	const Point& group(size_t index) const 
	{
		return *mGroups[index];
	}
	
	const cv::Rect2f& extension() const 
	{
		return mExtension;
	}
	
	const std::set<short>& angles() const 
	{
		return mAngles;
	}
	
	void createSampler(short minArcLength);
	
	Sampler* sampler() const 
	{
		return mSampler;
	}
	
	static const cv::Mat& edgeImg()  
	{
		return sEdgeImg; 
	}
	
	static std::vector<Point*>& points() 
	{
		return sPoints;
	}
	
private:
	static cv::Mat sEdgeImg;
	static std::vector<Point*> sPoints;
	std::vector<Point*> mPoints;
	std::vector<Point*> mGroups;
	short mNumAngles;
	cv::Rect2f mExtension;
	cv::Point2f mCenter;
	Sampler *mSampler;
	std::set<short> mAngles;
				
	PointCloud(int numAngles);
	
	void setExtension();
	
	static cv::Rect2f getExtension(const std::vector<PointCloud> &pointClouds);
	
};

#endif // _POINTCLOUD_H_
