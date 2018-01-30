#ifndef _HOUGH_CELL_H_
#define _HOUGH_CELL_H_

#include "houghaccumulator.h"

class HoughCell 
{
public: 
	HoughCell(const cv::Rect &extension, short minArcLength, float minCellSize, short branchingFactor, float maxIntersectionRatio = 1.5f);
	
	HoughCell(HoughCell *parent, short indX, short indY);
	
	~HoughCell();
	
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
		return mExtension.width;
	}
	
	short branchingFactor() const 
	{
		return mBranchingFactor;
	}
	
	short indX() const 
	{
		return mIndX;
	}
	
	short indY() const 
	{
		return mIndY;
	}
	
	const cv::Rect2f& extension() const 
	{
		return mExtension;
	}
	
	bool isVisited() const 
	{
		return mVisited;
	}
	
	const short minArcLength() const 
	{
		return mMinArcLength;
	}
	
	std::set<HoughAccumulator*> visit();
	
	void setVisited();
	
	HoughAccumulator* addIntersection(Sampler *sampler);
	
private:
	const short mMinArcLength;
	const float mMinCellSize;
	const short mBranchingFactor;
	const float mMaxIntersectionRatio;
	HoughCell *mParent;
	HoughCell **mChildren;
	cv::Rect2f mExtension;
	short mIndX, mIndY;
	bool mVisited;
	std::vector<HoughAccumulator*> mAccumulators;
		
	// reference: https://tavianator.com/fast-branchless-raybounding-box-intersections/
	bool pointIntersectsRect(const Point &p);
	
	bool intersectionBetweenPoints(Sampler *sampler, const std::pair<size_t, size_t> &sample, Intersection &intersection);

	short getChildIndex(const cv::Point2f &point, short &indX, short &indY)
	{
		cv::Point2f normalized = (point - mExtension.tl()) / mExtension.width;
		indX = static_cast<short>(normalized.x * mBranchingFactor);
		indY = static_cast<short>(normalized.y * mBranchingFactor);
		return indY * mBranchingFactor + indX;
	}

	HoughAccumulator* accumulate(const Intersection &intersection);
	
	HoughAccumulator* addIntersection(const Intersection &intersection);
	
};

#endif // _HOUGH_CELL_H_
