#include "houghcell.h"

#include <iostream>

HoughCell::HoughCell(const cv::Rect &extension, short minArcLength,
		short branchingFactor, float maxIntersectionRatio)
	: mMinArcLength(minArcLength)
	, mBranchingFactor(branchingFactor)
	, mMaxIntersectionRatio(maxIntersectionRatio)
	, mParent(this)
	, mExtension(extension)
	, mIndX(0)
	, mIndY(0)
	, mVisited(false)
{ 
	mChildren = (HoughCell**)calloc(branchingFactor * branchingFactor, sizeof(HoughCell*));
}

HoughCell::HoughCell(HoughCell *parent, short indX, short indY)
	: mMinArcLength(parent->mMinArcLength)
	, mBranchingFactor(parent->mBranchingFactor)
	, mMaxIntersectionRatio(parent->mMaxIntersectionRatio)
	, mParent(parent)
	, mIndX(indX)
	, mIndY(indY)
	, mVisited(false)
{
	float newSize = parent->mExtension.size().width / mBranchingFactor;
	float x = parent->mExtension.tl().x + indX * newSize;
	float y = parent->mExtension.tl().y + indY * newSize;
	mExtension = cv::Rect2f(x, y, newSize, newSize);
	mChildren = (HoughCell**)calloc(mBranchingFactor * mBranchingFactor, sizeof(HoughCell*));
}

HoughCell::~HoughCell()
{
	for (HoughAccumulator *accumulator : mAccumulators)
	{
		delete accumulator;
	}
	for (size_t i = 0, end = mBranchingFactor * mBranchingFactor; i < end; i++)
	{
		if (mChildren[i] != NULL)
			delete mChildren[i];
	}
	delete[] mChildren;
}
	
std::set<HoughAccumulator*> HoughCell::visit()
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif
	mVisited = true;
	std::set<HoughAccumulator*> accumulators;
	for (HoughAccumulator *accumulator : mAccumulators)
	{
		for (const Intersection &intersection : accumulator->intersections())
		{
			HoughAccumulator *childAccumulator = addIntersection(intersection);
			if (childAccumulator != NULL)
				accumulators.insert(childAccumulator);
		}
	}
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeVisit += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return accumulators;
}

void HoughCell::setVisited()
{
	mVisited = true;
}

HoughAccumulator* HoughCell::addIntersection(Sampler *sampler)
{
	std::pair<size_t, size_t> sample = sampler->sample();
	Intersection intersection;
	if (intersectionBetweenPoints(sampler, sample, intersection))
	{
		#ifdef _BENCHMARK
			auto begin = std::chrono::high_resolution_clock::now();
		#endif
		HoughAccumulator *accumulator = addIntersection(intersection);
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeAddIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return accumulator;
	}
	return NULL;
}

bool HoughCell::intersectionBetweenPoints(Sampler *sampler, const std::pair<size_t, size_t> &sample, Intersection &intersection)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif
	const Point &p1 = sampler->pointCloud().group(sample.first);
	const Point &p2 = sampler->pointCloud().group(sample.second);
	
	cv::Point2f a = p1.position;
	cv::Point2f b = p1.position + p1.normal;
	cv::Point2f c = p2.position;
	cv::Point2f d = p2.position + p2.normal;

	// Get (a, b, c) of the first line 
	float a1 = b.y - a.y;
	float b1 = a.x - b.x;
	float c1 = a1 * a.x + b1 * a.y;

	// Get (a, b, c) of the second line
	float a2 = d.y - c.y;
	float b2 = c.x - d.x;
	float c2 = a2 * c.x + b2 * c.y;

	// Get delta and check if the lines are parallel
	float delta = a1 * b2 - a2 * b1;
	if (std::abs(delta) < std::numeric_limits<float>::epsilon())
	{
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return false;
	}

	float x = (b2 * c1 - b1 * c2) / delta;
	float y = (a1 * c2 - a2 * c1) / delta;
	cv::Point2f position(x, y);
	
	// Check intersection ratio 
	float dist1 = norm(a - position);
	float dist2 = norm(c - position);
	if (dist1 < std::numeric_limits<float>::epsilon() || dist2 < std::numeric_limits<float>::epsilon() || 
		std::max(dist1, dist2) / std::min(dist1, dist2) > mMaxIntersectionRatio)
	{
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return false; 
	}
	
	intersection.sampler = sampler;
	intersection.p1 = sample.first;
	intersection.p2 = sample.second;
	intersection.position = position; 
	intersection.dist = (dist1 + dist2) / 2;
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return true;
}

HoughAccumulator* HoughCell::accumulate(const Intersection &intersection)
{
	HoughAccumulator *chosenAccumulator = NULL;
	for (HoughAccumulator *accumulator : mAccumulators)
	{
		if (std::abs(accumulator->radius() - intersection.dist) < std::max(10.0f, accumulator->radius() / 10))
		{
			chosenAccumulator = accumulator;
			break;
		}
	}
	if (chosenAccumulator == NULL)
	{
		chosenAccumulator = new HoughAccumulator(this);
		mAccumulators.push_back(chosenAccumulator);
	}
	chosenAccumulator->accumulate(intersection);
	return chosenAccumulator;
}

HoughAccumulator* HoughCell::addIntersection(const Intersection &intersection)
{
	if (mExtension.contains(intersection.position))
	{
		short indX, indY;
		short childIndex = getChildIndex(intersection.position, indX, indY);
		if (mChildren[childIndex] == NULL)
			mChildren[childIndex] = new HoughCell(this, indX, indY);
		if (mChildren[childIndex]->isVisited())
		{
			mChildren[childIndex]->addIntersection(intersection);
		}
		else
		{
			return mChildren[childIndex]->accumulate(intersection);
		}
	}
	else if (mParent != this)
	{
		return mParent->addIntersection(intersection);
	}
	return NULL;
}
	
