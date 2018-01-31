#ifndef _HOUGH_CELL_H_
#define _HOUGH_CELL_H_

#include "houghaccumulator.h"

class HoughCell 
{
public: 
	HoughCell(const cv::Rect &extension, short minArcLength, short branchingFactor, float maxIntersectionRatio = 1.5f);
	
	HoughCell(HoughCell *parent, short indX, short indY);
	
	~HoughCell();
	
	HoughCell* parent() const 
	{
		return mParent;
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
	
	const short minArcLength() const 
	{
		return mMinArcLength;
	}
	
	bool isVisited() const 
	{
		return mVisited;
	}
	
	std::set<HoughAccumulator*> visit();
	
	void setVisited();
	
	HoughAccumulator* addIntersection(Sampler *sampler);
	
private:
	const short mMinArcLength;
	const short mBranchingFactor;
	const float mMaxIntersectionRatio;
	HoughCell *mParent;
	HoughCell **mChildren;
	cv::Rect2f mExtension;
	short mIndX, mIndY;
	bool mVisited;
	std::vector<HoughAccumulator*> mAccumulators;
		
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
