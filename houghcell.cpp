#include "houghcell.h"

#include <iostream>

HoughCell::HoughCell(const cv::Rect2f &maxExtension, const cv::Point2f &center, float size, short numAngles, short minNumAngles)
	: mMaxExtension(maxExtension)
	, mCenter(center)
	, mSize(size)
	, mNumAngles(numAngles)
	, mMinNumAngles(minNumAngles)
	, mVisited(false)
	, mDepth(0)
	, mNumAccumulators((size_t)std::roundf(maxExtension.width / 2 / size))
{
	for (size_t i = 0; i < 4; i++)
	{
		mChildren[i] = NULL;
	}
	mAccumulators = new HoughAccumulator*[mNumAccumulators];
	for (size_t i = 0; i < mNumAccumulators; i++)
	{
		mAccumulators[i] = NULL;
	}
}

HoughCell::~HoughCell()
{
	for (size_t i = 0; i < mNumAccumulators; i++)
	{
		if (mAccumulators[i] != NULL)
			delete mAccumulators[i];
	}
	delete[] mAccumulators;
	for (size_t i = 0; i < 4; i++)
	{
		if (mChildren[i] != NULL)
			delete mChildren[i];
	}
}
	
std::set<HoughAccumulator*> HoughCell::addIntersectionsToChildren()
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif
	std::set<HoughAccumulator*> accumulators;
	for (size_t i = 0; i < mNumAccumulators; i++)
	{
		if (mAccumulators[i] != NULL)
		{
			for (const Intersection &intersection : mAccumulators[i]->intersections())
			{
				HoughAccumulator *childAccumulator = addIntersection(intersection);
				if (childAccumulator != NULL && childAccumulator->hasCircleCandidate())
				{
					accumulators.insert(childAccumulator);
				}
			}
		}
	}
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeAddIntersectionsChildren += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return accumulators;
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
	
	// Check if intersection falls on valid range 
	if (!mMaxExtension.contains(position))
	{
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return false;
	}
	
	// Check intersection ratio 
	float dist1 = norm(a - position);
	float dist2 = norm(c - position);
	if (dist1 < std::numeric_limits<float>::epsilon() || dist2 < std::numeric_limits<float>::epsilon() || 
		std::max(dist1, dist2) / std::min(dist1, dist2) > MAX_INTERSECTION_RATIO)
	{
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return false; 
	}
	
	// Check ratio between dist of points and intersection dist 
	float dist = (dist1 + dist2) / 2;
	float distPoints = norm(a - c);
	if (std::max(dist, distPoints) / std::min(dist, distPoints) > 10 * MAX_INTERSECTION_RATIO)
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
	intersection.dist = dist;
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return true;
}

HoughAccumulator* HoughCell::accumulate(const Intersection &intersection)
{
	size_t radius = std::min(mNumAccumulators - 1, (size_t)(intersection.dist / mSize));
	if (mAccumulators[radius] == NULL)
	{
		mAccumulators[radius] = new HoughAccumulator(this, radius * mSize + mSize / 2);
	}
	if (!mAccumulators[radius]->isVisited())
	{
		mAccumulators[radius]->accumulate(intersection);
		return mAccumulators[radius];
	}
	return NULL;
}

HoughAccumulator* HoughCell::addIntersection(const Intersection &intersection)
{
	if (!mVisited)
	{
		return accumulate(intersection);
	}
	else
	{
		int childIndex = ((intersection.position.x > mCenter.x) << 1) | (intersection.position.y > mCenter.y);
		if (mChildren[childIndex] != NULL)
		{
			return mChildren[childIndex]->addIntersection(intersection);
		}
		else
		{
			float size = mSize / 2;
			cv::Point2f center;
			switch (childIndex)
			{
			case 0:
				center = cv::Point2f(mCenter.x - size, mCenter.y - size);
				break;
			case 1:
				center = cv::Point2f(mCenter.x - size, mCenter.y + size);
				break;
			case 2:
				center = cv::Point2f(mCenter.x + size, mCenter.y - size);
				break;
			case 3:
				center = cv::Point2f(mCenter.x + size, mCenter.y + size);
				break;
			}
			mChildren[childIndex] = new HoughCell(mMaxExtension, center, size, mNumAngles, mMinNumAngles);
			mChildren[childIndex]->mDepth = mDepth + 1;
			return mChildren[childIndex]->accumulate(intersection);
		}
	}
}
	
