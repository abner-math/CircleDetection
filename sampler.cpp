#include "sampler.h"

#include <iostream>

std::default_random_engine Sampler::sRandomGenerator;
std::gamma_distribution<double> Sampler::sGammaDistribution = std::gamma_distribution<double>(1, 1);
double Sampler::sGammaThreshold = 3.0;

Sampler::Sampler(const Quadtree *quadtree, float climbChance, size_t minArcLength)
	: mParent(this)
	, mQuadtree(quadtree)
	, mClimbChance(climbChance)
	, mMinArcLength(minArcLength)
	, mPoints(std::vector<size_t>(numPoints()))
	, mNumPointsPerAngle(std::vector<size_t>(numAngles(), 0))
	, mNumEmptyAngles(numAngles())
	, mNumAvailablePoints(numPoints())
{
	for (size_t i = 0; i < numPoints(); i++)
	{
		mPoints[i] = i;
		if (++mNumPointsPerAngle[normalAngleIndex(i)] == 1)
			--mNumEmptyAngles;
	}
}

Sampler::Sampler(Sampler &parent)
	: mParent(&parent)
	, mQuadtree(parent.mQuadtree)
	, mClimbChance(parent.mClimbChance)
	, mMinArcLength(parent.mMinArcLength)
	, mPoints(parent.mPoints)
	, mNumPointsPerAngle(parent.mNumPointsPerAngle)
	, mNumEmptyAngles(parent.mNumEmptyAngles)
	, mNumAvailablePoints(parent.mNumAvailablePoints)
{

}

bool Sampler::canSample() const 
{
	return (numAngles() - mNumEmptyAngles) > mMinArcLength;
}

std::pair<size_t, size_t> Sampler::sample(const std::set<size_t> &points) const 
{
	std::pair<size_t, size_t> p;
	p.first = selectRandomPoint(points);
	p.second = selectAnotherRandomPoint(p.first);
	return p;
}

bool Sampler::isRemoved(size_t point) const 
{
	return mPoints[point] != point;
}

void Sampler::removePoint(size_t point)
{
	if (isRemoved(point)) return;
	mPoints[point] = selectSubstitutePoint(point);
	if (--mNumPointsPerAngle[normalAngleIndex(point)] == 0)
		++mNumEmptyAngles;
	--mNumAvailablePoints;
}

void Sampler::removePointFromAll(size_t point)
{
	removePoint(point);
	if (mParent != this)
		mParent->removePointFromAll(point);
}

size_t Sampler::getPoint(size_t index) const 
{
	if (index == mPoints[index]) return index;
	return getPoint(mPoints[index]);
}

size_t Sampler::selectRandomPoint(const std::set<size_t> &points) const 
{
	if (points.empty() || getProbability() < mClimbChance)
	{
		double value = std::min(sGammaDistribution(sRandomGenerator), sGammaThreshold) / (sGammaThreshold + 0.001);
		return getPoint(static_cast<size_t>(value * mPoints.size()));
	}
	else
	{
		double value = std::min(sGammaDistribution(sRandomGenerator), sGammaThreshold) / (sGammaThreshold + 0.001);
		size_t index = static_cast<size_t>(value * points.size());
		auto it = points.begin();
		std::advance(it, index);
		return getPoint(*it);
	}
}

bool Sampler::containsValidPoints(const Quadtree *node, size_t firstPoint, size_t angle) const
{
	for (const size_t &point : node->points(angle))
	{
		if (point != firstPoint && !isRemoved(point))
			return true;
	}
	return false;
}

size_t Sampler::resolveAngle(const Quadtree *node, size_t firstPoint, size_t angle) const
{
	if (containsValidPoints(node, firstPoint, angle)) return angle;
	size_t clockwiseAngle = angle;
	size_t counterClockwisewAngle = angle;
	for (size_t i = 0; i < numAngles() / 2; i++)
	{
		clockwiseAngle = increaseOneAngle(clockwiseAngle);
		if (containsValidPoints(node, firstPoint, clockwiseAngle)) return clockwiseAngle;
		counterClockwisewAngle = decreaseOneAngle(counterClockwisewAngle);
		if (containsValidPoints(node, firstPoint, counterClockwisewAngle)) return counterClockwisewAngle;
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::selectSubstitutePoint(size_t firstPoint) const 
{
	const Quadtree *node = mQuadtree->findLeaf(firstPoint); 
	size_t firstAngle = rand() % numAngles();//normalAngleIndex(firstPoint);
	size_t secondAngle;
	while (true)
	{
		secondAngle = resolveAngle(node, firstPoint, firstAngle);
		if (secondAngle > numAngles())
		{
			node = node->parent();
		}
		else
		{
			break;
		}
	}
	for (const size_t &point : node->points(secondAngle))
	{
		if (point != firstPoint && !isRemoved(point))
			return point;
	}
	return std::numeric_limits<size_t>::max();
}

size_t Sampler::selectAnotherRandomPoint(size_t firstPoint) const 
{
	const Quadtree *node = mQuadtree->findLeaf(firstPoint); 
	while (!node->isRoot() && getProbability() < mClimbChance)
	{
		node = node->parent();
	}
	size_t firstAngle = normalAngleIndex(firstPoint);
	size_t secondAngle;
	while (true)
	{
		std::vector<size_t> angles;
		angles.reserve(numAngles() * numAngles());
		for (size_t i = 0; i < numAngles(); i++)
		{
			size_t diff = i > firstAngle ? i - firstAngle : firstAngle - i;
			diff = std::min(diff, numAngles() - diff);
			for (size_t j = 0; j < diff; j++)
			{
				angles.push_back(i);
			}
		}
		secondAngle = resolveAngle(node, firstPoint, angles[rand() % angles.size()]);
		if (secondAngle > numAngles())
		{
			node = node->parent();
		}
		else
		{
			break;
		}
	}
	return getPoint(node->points(secondAngle)[rand() % node->points(secondAngle).size()]);
}

