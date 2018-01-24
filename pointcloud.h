#ifndef _POINTCLOUD_H_
#define _POINTCLOUD_H_

#include <opencv2/core/core.hpp>

struct Point
{
	cv::Point2f position;
	cv::Point2f normal;
	cv::Point2f inverseNormal;
	double curvature;
	double normalAngle;
	size_t normalAngleIndex;
};

class ImageUtils
{
public:
	ImageUtils(const cv::Mat &img)
		: mImg(img)
	{
		
	}
	
	int valueAt(int x, int y) const   
	{
		const uchar *ptr = (uchar*)mImg.data;
		return static_cast<int>(ptr[y * mImg.cols + x]);
	}

	int sobelX(int x, int y) const 
	{
		return valueAt(x - 1, y - 1) - valueAt(x + 1, y - 1) +
				2 * valueAt(x - 1, y) - 2 * valueAt(x + 1, y) +
				valueAt(x - 1, y + 1) - valueAt(x + 1, y + 1);
	}

	int sobelY(int x, int y) const 
	{
		return valueAt(x - 1, y - 1) - valueAt(x - 1, y + 1) +
				2 * valueAt(x, y - 1) - 2 * valueAt(x, y + 1) +
				valueAt(x + 1, y - 1) - valueAt(x + 1, y + 1);
	}

private:
	const cv::Mat &mImg;
	
};

inline float norm(const cv::Point2f &p)
{
	return std::sqrt(p.x * p.x + p.y * p.y);
}

class PointCloud
{
public:
	PointCloud(const cv::Mat &gray, const cv::Mat &edges, size_t numAngles);

	~PointCloud();
	
	size_t numAngles() const 
	{
		return mNumAngles;
	}
	
	size_t numPoints() const 
	{
		return mPoints.size();
	}
	
	const Point* point(size_t index) const 
	{
		return mPoints[index];
	}
	
	const cv::Rect2f& extension() const 
	{
		return mRect;
	}
	
private:
	std::vector<Point*> mPoints;
	cv::Rect2f mRect;
	size_t mNumAngles;
				
	inline size_t getNormalAngleIndex(double normalAngle)
	{
		return static_cast<size_t>(std::round(normalAngle / (180.0 / mNumAngles))) % mNumAngles;
	}
	
	double getNormalAngle(const cv::Point2f &normal);

	void setExtension();
	
	void sortPointsByCurvature();
	
	static double calculateCurvature(const std::vector<Point*> &points, int x, int y, int cols);

};

#endif // _POINTCLOUD_H_
