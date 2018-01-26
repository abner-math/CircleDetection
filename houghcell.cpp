#include "houghcell.h"

#include <iostream>

HoughCell::HoughCell(Sampler *sampler, size_t branchingFactor, float minCellSize,
			float maxIntersectionRatio)
	: mPointCloud(sampler->quadtree()->pointCloud())
	, mBranchingFactor(branchingFactor)
	, mMinCellSize(minCellSize)
	, mMaxIntersectionRatio(maxIntersectionRatio)
	, mParent(this)
	, mChildren(std::vector<HoughCell*>(branchingFactor * branchingFactor, NULL))
	, mSampler(sampler)
	, mRect(sampler->quadtree()->pointCloud()->extension())
	, mIndX(0)
	, mIndY(0)
	, mVisited(false)
{ 
	
}

HoughCell::HoughCell(HoughCell *parent, size_t indX, size_t indY)
	: mPointCloud(parent->mPointCloud)
	, mBranchingFactor(parent->mBranchingFactor)
	, mMinCellSize(parent->mMinCellSize)
	, mMaxIntersectionRatio(parent->mMaxIntersectionRatio)
	, mParent(parent)
	, mChildren(std::vector<HoughCell*>(mBranchingFactor * mBranchingFactor, NULL))
	, mSampler(NULL)
	, mIndX(indX)
	, mIndY(indY)
	, mVisited(false)
{
	float newSize = parent->mRect.size().width / mBranchingFactor;
	float x = parent->mRect.tl().x + indX * newSize;
	float y = parent->mRect.tl().y + indY * newSize;
	mRect = cv::Rect2f(x, y, newSize, newSize);
}

HoughCell::~HoughCell()
{
	for (HoughAccumulator *accumulator : mAccumulators)
	{
		delete accumulator;
	}
	if (mSampler != NULL)
		delete mSampler;
	for (size_t i = 0, end = mBranchingFactor * mBranchingFactor; i < end; i++)
	{
		if (mChildren[i] != NULL)
			delete mChildren[i];
	}
}

std::set<HoughAccumulator*> HoughCell::visit()
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif
	mVisited = true;
	mSampler = new Sampler(*mParent->mSampler);
	for (size_t i = 0; i < mPointCloud->numPoints(); i++)
	{
		if (!mSampler->isRemoved(i) && !pointIntersectsRect(mPointCloud->point(i)))
			mSampler->removePoint(i);
	}
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

HoughAccumulator* HoughCell::addIntersection()
{
	std::pair<size_t, size_t> sample = mSampler->sample();
	Intersection intersection;
	if (intersectionBetweenPoints(sample, intersection))
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


bool HoughCell::pointIntersectsRect(const Point &p)
{
	float tx1 = (mRect.tl().x - p.position.x) * p.inverseNormal.x;
	float tx2 = (mRect.br().x - p.position.x) * p.inverseNormal.x;
 
	float tmin = std::min(tx1, tx2);
	float tmax = std::max(tx1, tx2);
 
	float ty1 = (mRect.tl().y - p.position.y) * p.inverseNormal.y;
	float ty2 = (mRect.br().y - p.position.y) * p.inverseNormal.y;
 
	tmin = std::max(tmin, std::min(ty1, ty2));
	tmax = std::min(tmax, std::max(ty1, ty2));
	
	return tmax >= tmin;
}

bool HoughCell::intersectionBetweenPoints(const std::pair<size_t, size_t> &sample, Intersection &intersection)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif
	const Point &p1 = mPointCloud->point(sample.first);
	const Point &p2 = mPointCloud->point(sample.second);
	
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
		if (std::abs(accumulator->radius() - intersection.dist) < mMinCellSize)
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
	if (mRect.contains(intersection.position))
	{
		size_t indX, indY;
		size_t childIndex = getChildIndex(intersection.position, indX, indY);
		if (mChildren[childIndex] == NULL)
			mChildren[childIndex] = new HoughCell(this, indX, indY);
		if (!mChildren[childIndex]->isVisited())
		{
			mSampler->removePointFromAll(intersection.p1);
			mSampler->removePointFromAll(intersection.p2);
			return mChildren[childIndex]->accumulate(intersection);
		}
	}
	else if (mParent != this)
	{
		//mSampler->removePoint(intersection.p1);
		//mSampler->removePoint(intersection.p2);
		return mParent->addIntersection(intersection);
	}
	return NULL;
}
	
