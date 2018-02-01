#ifndef _HOUGH_CELL_H_
#define _HOUGH_CELL_H_

#include "houghaccumulator.h"

#define MAX_INTERSECTION_RATIO 1.5f

class HoughCell 
{
public: 
	HoughCell(const cv::Rect2f &maxExtension, const cv::Point2f &center, float size, short numAngles, short minNumAngles);
	
	~HoughCell();
	
	const HoughCell* child(size_t index) const 
	{
		return mChildren[index];
	}
	
	size_t numAccumulators() const 
	{
		return mNumAccumulators;
	}
	
	const HoughAccumulator* accumulator(size_t index) const 
	{
		return mAccumulators[index];
	}
	
	const cv::Rect2f& maxExtension() const 
	{
		return mMaxExtension;
	}
	
	cv::Rect2f extension() const 
	{
		return cv::Rect2f(mCenter.x - mSize, mCenter.y - mSize, mSize * 2, mSize * 2);
	}
	
	const cv::Point2f& center() const 
	{
		return mCenter;
	}
	
	float size() const
	{
		return mSize;
	}
	
	short numAngles() const 
	{
		return mNumAngles;
	}
	
	short minNumAngles() const 
	{
		return mMinNumAngles;
	}
	
	size_t depth() const 
	{
		return mDepth;
	}
	
	bool isVisited() const 
	{
		return mVisited;
	}
	
	void setVisited()
	{
		mVisited = true;
	}
	
	std::set<HoughAccumulator*> addIntersectionsToChildren();
	
	HoughAccumulator* addIntersection(Sampler *sampler);
	
private:
	HoughCell *mChildren[4];
	cv::Rect2f mMaxExtension;
	cv::Point2f mCenter;
	float mSize;
	short mNumAngles;
	short mMinNumAngles;
	bool mVisited;
	size_t mNumAccumulators;
	size_t mDepth;
	HoughAccumulator **mAccumulators;
		
	bool intersectionBetweenPoints(Sampler *sampler, const std::pair<size_t, size_t> &sample, Intersection &intersection);

	HoughAccumulator* accumulate(const Intersection &intersection);
	
	HoughAccumulator* addIntersection(const Intersection &intersection);
	
};

#endif // _HOUGH_CELL_H_
