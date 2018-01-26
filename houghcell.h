#ifndef _HOUGH_CELL_H_
#define _HOUGH_CELL_H_

#include "houghaccumulator.h"

class HoughCell 
{
public: 
	HoughCell(Sampler *sampler, size_t branchingFactor, float minCellSize,
				float maxIntersectionRatio);
	
	HoughCell(HoughCell *parent, size_t indX, size_t indY);
	
	~HoughCell();
	
	const PointCloud* pointCloud() const 
	{
		return mPointCloud;
	}
	
	HoughCell* parent() const 
	{
		return mParent;
	}
	
	float minCellSize() const 
	{
		return mMinCellSize;
	}
	
	float cellSize() const 
	{
		return mRect.width;
	}
	
	size_t indX() const 
	{
		return mIndX;
	}
	
	size_t indY() const 
	{
		return mIndY;
	}
	
	const cv::Rect2f& rect() const 
	{
		return mRect;
	}
	
	bool isVisited() const 
	{
		return mVisited;
	}
	
	const size_t minArcLength() const 
	{
		return mParent->mSampler->minArcLength();
	}
	
	bool isTermined() const 
	{
		return !mSampler->canSample();
	}
	
	Sampler* sampler()  
	{
		return mSampler;
	}
	
	std::set<HoughAccumulator*> visit();
	
	void setVisited();
	
	HoughAccumulator* addIntersection();
	
private:
	const PointCloud *mPointCloud;
	const size_t mBranchingFactor;
	const float mMinCellSize;
	const float mMaxIntersectionRatio;
	HoughCell *mParent;
	std::vector<HoughCell*> mChildren;
	Sampler *mSampler;
	cv::Rect2f mRect;
	size_t mIndX, mIndY;
	bool mVisited;
	std::vector<HoughAccumulator*> mAccumulators;
	std::vector<size_t> mPoints;
		
	// reference: https://tavianator.com/fast-branchless-raybounding-box-intersections/
	bool pointIntersectsRect(const Point &p);
	
	bool intersectionBetweenPoints(const std::pair<size_t, size_t> &sample, Intersection &intersection);

	inline size_t getChildIndex(const cv::Point2f &point, size_t &indX, size_t &indY)
	{
		cv::Point2f normalized = (point - mRect.tl()) / mRect.width;
		indX = static_cast<size_t>(normalized.x * mBranchingFactor);
		indY = static_cast<size_t>(normalized.y * mBranchingFactor);
		return indY * mBranchingFactor + indX;
	}

	HoughAccumulator* accumulate(const Intersection &intersection);
	
	HoughAccumulator* addIntersection(const Intersection &intersection);
	
};

#endif // _HOUGH_CELL_H_
