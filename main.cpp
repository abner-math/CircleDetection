//#define _BENCHMARK_

#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <list>
#include <set>
#include <chrono>
#include <memory>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>

// pre-processing parameters
#define CANNY_MAX_THRESHOLD 100
#define CANNY_RATIO 3
#define CANNY_KERNEL_SIZE 3

// algorithm parameters 
#define COMPONENTS_GRADIENT_THRESHOLD 0.85f 
#define HOUGH_INITIAL_NUM_BINS 2 
#define ACCUMULATOR_NUM_ANGLES 36
#define ACCUMULATOR_NUM_RADIUS 10
#define ACCUMULATOR_MIN_ARC_LENGTH 9 

std::string gEdgeWindowName = "Edge image";
int gCannyLowThreshold;
int gCirclePrecision;
cv::Mat gImg;
cv::Mat gImgGray;

std::default_random_engine gRandomGenerator;
std::gamma_distribution<double> gGammaDistribution(3, 1);
double gGammaThreshold = 9.0;

#ifdef _BENCHMARK_
	double benchmarkInitialization = 0;
	double benchmarkSample = 0;
	double benchmarkIntersection = 0;
	double benchmarkAddIntersection = 0;
	double benchmarkCreateComponent = 0;
#endif

class Component;

struct Point
{
	Point()
	{
		component = NULL;
		marked = false;
		for (int i = 0; i < 8; i++)
			neighbors[i] = NULL;
	}
	
	cv::Point2f position;
	cv::Point2f normal;
	cv::Point2f inverseNormal;
	short curvature;
	size_t normalAngle;
	Point *neighbors[8];
	Component *component;
	bool marked;
};

class Component
{
public:
	~Component()
	{
		for (Point *point : mPoints)
		{
			point->component = NULL;
		}
	}
	
	void addPoint(Point *point)
	{
		point->component = this;
		mPoints.push_back(point);
	}
	
	size_t numPoints() const 
	{
		return mPoints.size();
	}
	
	const Point* point(size_t index) const 
	{
		return mPoints[index];
	}
	
	const cv::Rect2f& rect() const 
	{
		return mRect;
	}
	
	void calculateRect()
	{
		float minX, minY, maxX, maxY;
		minX = minY = std::numeric_limits<float>::max();
		maxX = maxY = -std::numeric_limits<float>::max();
		for (Point *point : mPoints)
		{
			float x = point->position.x;
			float y = point->position.y;
			if (x < minX) minX = x;
			if (x > maxX) maxX = x;
			if (y < minY) minY = y;
			if (y > maxY) maxY = y;
		}
		float size = std::max(maxX - minX, maxY - minY);
		float centerX = (minX + maxX) / 2;
		float centerY = (minY + maxY) / 2;
		minX = centerX - size;
		minY = centerY - size;
		mRect = cv::Rect2f(minX, minY, 2*size, 2*size); 
	}
	
	void reorientNormals()
	{
		cv::Point2f viewpoint = mRect.tl() + cv::Point2f(mRect.width, mRect.height) / 2.0f;
		for (Point *point : mPoints)
		{
			reorientNormalTowardsViewpoint(point, viewpoint);
			calculateNormalAngle(point);
		}
	}
	
private:
	std::vector<Point*> mPoints;
	cv::Rect2f mRect;
		
	inline void reorientNormalTowardsViewpoint(Point *point, cv::Point2f &viewpoint)
	{
		if (point->normal.dot(point->position - viewpoint) < 0)
			point->normal = -point->normal;
		point->inverseNormal = cv::Point2f(1 / point->normal.x, 1 / point->normal.y);
	}

	inline void calculateNormalAngle(Point *point)
	{
		double angleRadians = std::atan2(point->normal.y, point->normal.x) + M_PI;
		double angleDegrees = std::max(0.0, angleRadians * 180 / M_PI);
		point->normalAngle = static_cast<size_t>(std::round(angleDegrees / (360.0 / ACCUMULATOR_NUM_ANGLES))) % ACCUMULATOR_NUM_ANGLES;
	}

};

struct Circle
{
	cv::Point2f center;
	float radius;
};

struct Intersection
{
	const Point *p1;
	const Point *p2;
	cv::Point2f position;
	float dist;
	size_t radiusBin;
};


// Generic functor
template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    const int mInputs, mValues;

    Functor() : mInputs(InputsAtCompileTime), mValues(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : mInputs(inputs), mValues(values) {}

    int inputs() const { return mInputs; }
    int values() const { return mValues; }
};


class CircleFunctor : public Functor<float>
{
public:
    CircleFunctor(const Eigen::Matrix2Xf &points)
        : Functor<float>(2, points.cols())
        , mPoints(points)
    {

    }

    int operator()(const Eigen::VectorXf &params, Eigen::VectorXf &residuals)
    {
        Eigen::Vector2f center(params(0), params(1));
        float radius = std::abs(params(2));
        float radiusSquared = radius * radius;
        residuals = ((mPoints.colwise() - center).colwise().squaredNorm().array() - radiusSquared).pow(2);
        return 0;
    }

    int df(const Eigen::VectorXf &params, Eigen::MatrixXf &jacobian)
    {
        Eigen::Vector2f center(params(0), params(1));
        float radius = std::abs(params(2));
        float radiusSquared = radius * radius;
        Eigen::Matrix2Xf pointsMinusCenter = mPoints.colwise() - center;
        Eigen::Array<float, 1, -1> val = pointsMinusCenter.colwise().squaredNorm().array() - radiusSquared;
        jacobian.col(0) = -4 * pointsMinusCenter.row(0).array() * val;
        jacobian.col(1) = -4 * pointsMinusCenter.row(1).array() * val;
        jacobian.col(2) = -4 * radius * val;
        return 0;
    }

private:
    Eigen::Matrix2Xf mPoints;

};
class Wheel
{
public:
	Wheel()
		: mNumAngles(0)
	{
		for (size_t i = 0; i < ACCUMULATOR_NUM_ANGLES; i++)
		{
			mWheel[i] = false;
		}
	}
	
	~Wheel()
	{
		for (auto it = mIntersections.begin(); it != mIntersections.end(); ++it)
		{
			it->reset();
		}
	}
	
	bool addIntersection(const std::shared_ptr<const Intersection> &i)
	{
		mIntersections.push_back(i);
		if (!mWheel[i->p1->normalAngle])
		{
			++mNumAngles;
			mWheel[i->p1->normalAngle] = true;
		}
		if (!mWheel[i->p2->normalAngle])
		{
			++mNumAngles;
			mWheel[i->p2->normalAngle] = true;
		}
		return mNumAngles >= ACCUMULATOR_MIN_ARC_LENGTH;
	}
	
	size_t numIntersections() const 
	{
		return mIntersections.size();
	}
	
	std::shared_ptr<const Intersection> intersection(size_t index) const 
	{
		return mIntersections[index];
	}
	
	Circle* getCircle() const 
	{
		Circle *circle = new Circle;
		std::vector<float> xs, ys, dists;
		for (const std::shared_ptr<const Intersection> &intersection : mIntersections)
		{
			xs.push_back(intersection->position.x);
			ys.push_back(intersection->position.y);
			dists.push_back(intersection->dist);
		}
		size_t median = mIntersections.size() / 2;
		// center = median of positions 
		std::nth_element(xs.begin(), xs.begin() + median, xs.end());
		std::nth_element(ys.begin(), ys.begin() + median, ys.end());
		circle->center = cv::Point2f(xs[median], ys[median]);
		// radius = median of dists 
		std::nth_element(dists.begin(), dists.begin() + median, dists.end());
		circle->radius = dists[median];
		// least squares 
		if (mIntersections.size() > 4)
		{
			int col = 0;
			Eigen::Matrix2Xf points(2, mIntersections.size() * 2);
			for (const std::shared_ptr<const Intersection> &intersection : mIntersections)
			{
				points.col(col++) = Eigen::Vector2f(intersection->p1->position.x, intersection->p1->position.y);
				points.col(col++) = Eigen::Vector2f(intersection->p2->position.x, intersection->p2->position.y);
			}
			Eigen::VectorXf params(3);
			params << circle->center.x, circle->center.y, circle->radius;
			CircleFunctor functor(points);
			Eigen::LevenbergMarquardt<CircleFunctor, float> lm(functor);
			lm.minimize(params);
			circle->center = cv::Point2f(params(0), params(1));
			circle->radius = std::abs(params(2));
		}
		return circle;
	}
	
private:
	bool mWheel[ACCUMULATOR_NUM_ANGLES];
	size_t mNumAngles;
	std::vector<std::shared_ptr<const Intersection> > mIntersections;
	
};

class Accumulator
{
public:
	Accumulator()
		: mVisited(false)
	{
		for (size_t i = 0; i < ACCUMULATOR_NUM_RADIUS; i++)
		{
			mWheels[i] = NULL;
		}
	}
	
	~Accumulator()
	{
		for (size_t i = 0; i < ACCUMULATOR_NUM_RADIUS; i++)
		{
			if (mWheels[i] != NULL)
			{
				delete mWheels[i];
			}
		}
	}
	
	bool visited() const 
	{
		return mVisited;
	}
	
	void visited(bool visited)
	{
		mVisited = visited;
	}
	
	const Wheel* addIntersection(const std::shared_ptr<const Intersection> &i)
	{
		#ifdef _BENCHMARK_
			auto begin = std::chrono::high_resolution_clock::now();
		#endif 
		if (mWheels[i->radiusBin] == NULL) mWheels[i->radiusBin] = new Wheel;
		if (mWheels[i->radiusBin]->addIntersection(i))
		{
			#ifdef _BENCHMARK_
				auto end = std::chrono::high_resolution_clock::now();
				benchmarkAddIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
			#endif 
			return mWheels[i->radiusBin];
		}
		#ifdef _BENCHMARK_
			auto end = std::chrono::high_resolution_clock::now();
			benchmarkAddIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return NULL;
	}
	
private:
	Wheel *mWheels[ACCUMULATOR_NUM_RADIUS];
	bool mVisited;
	
};

class Sampler
{
public:
	Sampler(const Component *component)
		: mComponent(component)
		, mNumEmptyAngles(ACCUMULATOR_NUM_ANGLES)
	{
		for (size_t i = 0; i < ACCUMULATOR_NUM_ANGLES; i++)
		{
			mNumPointsPerAngle[i] = 0;
		}
		mPoints.resize(component->numPoints());
		for (size_t i = 0; i < component->numPoints(); i++)
		{
			mPoints[i] = i;
			size_t angle = mComponent->point(i)->normalAngle;
			if (++mNumPointsPerAngle[angle] == 1)
				--mNumEmptyAngles;
		}
	}
	
	bool canSample() const 
	{
		return (ACCUMULATOR_NUM_ANGLES - mNumEmptyAngles) > ACCUMULATOR_MIN_ARC_LENGTH;
	}
	
	std::pair<size_t, size_t> sample() const 
	{
		#ifdef _BENCHMARK_
			auto begin = std::chrono::high_resolution_clock::now();
		#endif 
		std::pair<size_t, size_t> p;
		p.first = selectRandomPoint();
		p.second = selectAnotherRandomPoint(p.first);
		#ifdef _BENCHMARK_
			auto end = std::chrono::high_resolution_clock::now();
			benchmarkSample += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return p;
	}
	
	bool isRemoved(size_t point) const 
	{
		mPoints[point] != point;
	}
	
	void removePoint(size_t point)
	{
		#ifdef _BENCHMARK_
			auto begin = std::chrono::high_resolution_clock::now();
		#endif 
		if (point == mPoints.size() - 1)
		{
			mPoints[point] = mPoints[0];
		}
		else
		{
			mPoints[point] = mPoints[point + 1];
		}
		int x = point - 1;
		while (x != point && mPoints[x] == point)
		{
			mPoints[x] = mPoints[point];
			if (--x < 0) x = mPoints.size() - 1;
		}
		if (--mNumPointsPerAngle[mComponent->point(point)->normalAngle] == 0)
			++mNumEmptyAngles;
		#ifdef _BENCHMARK_
			auto end = std::chrono::high_resolution_clock::now();
			benchmarkSample += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
	}
	
private:
	const Component *mComponent;
	std::vector<size_t> mPoints;
	size_t mNumPointsPerAngle[ACCUMULATOR_NUM_ANGLES];
	size_t mNumEmptyAngles;
	
	inline size_t selectRandomPoint() const 
	{
		size_t index = rand() % mPoints.size();
		return mPoints[index];
	}
	
	inline size_t selectAnotherRandomPoint(size_t point) const 
	{
		size_t anotherPoint = mPoints[rand() % mPoints.size()];
		while (true)
		{
			int angle = angleBetween(point, anotherPoint);
			if (angle != 0 && angle != 180) break;
			if (++anotherPoint == mPoints.size()) anotherPoint = 0;
			anotherPoint = mPoints[anotherPoint];
		}
		return anotherPoint;
	}
	
	int angleBetween(size_t pointA, size_t pointB) const 
	{
		int angleA = mComponent->point(pointA)->normalAngle * (360 / ACCUMULATOR_NUM_ANGLES);
		int angleB = mComponent->point(pointB)->normalAngle * (360 / ACCUMULATOR_NUM_ANGLES);
		int diff = angleA - angleB;
		return std::abs((diff + 180) % 360 - 180);
	}
	
};

std::shared_ptr<Intersection> intersectionBetweenPoints(const Point *p1, const Point *p2)
{
	#ifdef _BENCHMARK_
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	cv::Point2f a = p1->position;
	cv::Point2f b = p1->position + p1->normal;
	cv::Point2f c = p2->position;
	cv::Point2f d = p2->position + p2->normal;

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
		#ifdef _BENCHMARK_
			auto end = std::chrono::high_resolution_clock::now();
			benchmarkIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return NULL;
	}

	float x = (b2 * c1 - b1 * c2) / delta;
	float y = (a1 * c2 - a2 * c1) / delta;
	
	std::shared_ptr<Intersection> i = std::make_shared<Intersection>();
	i->p1 = p1;
	i->p2 = p2;
	i->position = cv::Point2f(x, y);
	cv::Point2f diff = p1->position - i->position;
	i->dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);
	#ifdef _BENCHMARK_
		auto end = std::chrono::high_resolution_clock::now();
		benchmarkIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	return i; 
}

// reference: https://tavianator.com/fast-branchless-raybounding-box-intersections/
bool pointIntersectsRect(const Point *p, const cv::Rect2f &r)
{
	float tx1 = (r.tl().x - p->position.x) * p->inverseNormal.x;
    float tx2 = (r.br().x - p->position.x) * p->inverseNormal.x;
 
    float tmin = std::min(tx1, tx2);
    float tmax = std::max(tx1, tx2);
 
    float ty1 = (r.tl().y - p->position.y) * p->inverseNormal.y;
    float ty2 = (r.br().y - p->position.y) * p->inverseNormal.y;
 
    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));
 
    return tmax >= tmin;
}

inline void calculateAccumulatorBin(const cv::Point2f &point, const cv::Rect2f &rect, size_t &binX, size_t &binY)
{
	cv::Point2f normalized = (point - rect.tl()) / rect.size().width;
	binX = static_cast<size_t>(normalized.x * HOUGH_INITIAL_NUM_BINS);
	binY = static_cast<size_t>(normalized.y * HOUGH_INITIAL_NUM_BINS);
}

inline size_t calculateRadiusBin(float dist, float maxDist)
{
	return static_cast<size_t>(dist / maxDist * ACCUMULATOR_NUM_RADIUS);	
}

void findCircles(const Component *component, Sampler &sampler, const cv::Rect2f &rect, std::vector<Circle*> &circles, const Wheel *parentWheel = NULL)
{
	#ifdef _BENCHMARK_
		auto begin = std::chrono::high_resolution_clock::now();
	#endif 
	if (!sampler.canSample())
	{
		#ifdef _BENCHMARK_
			auto end = std::chrono::high_resolution_clock::now();
			benchmarkInitialization += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		return;
	}
	Accumulator *accumulators = new Accumulator[HOUGH_INITIAL_NUM_BINS * HOUGH_INITIAL_NUM_BINS];	
	#ifdef _BENCHMARK_
		auto end = std::chrono::high_resolution_clock::now();
		benchmarkInitialization += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
	// rearrange intersections from parent wheel
	if (parentWheel != NULL)
	{
		std::map<const Wheel*, size_t> wheels;
		for (size_t i = 0; i < parentWheel->numIntersections(); i++)
		{
			#ifdef _BENCHMARK_
				begin = std::chrono::high_resolution_clock::now();
			#endif 
			size_t binX, binY;
			calculateAccumulatorBin(parentWheel->intersection(i)->position, rect, binX, binY);
			size_t index = binY * HOUGH_INITIAL_NUM_BINS + binX;
			const Wheel *wheel = accumulators[index].addIntersection(parentWheel->intersection(i));
			#ifdef _BENCHMARK_
				end = std::chrono::high_resolution_clock::now();
				benchmarkIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
			#endif 
			if (wheel != NULL)
			{
				wheels[wheel] = index;
			}
		}
		for (const std::pair<const Wheel*, size_t> &p : wheels)
		{
			size_t index = p.second;
			if (!accumulators[index].visited())
			{
				accumulators[index].visited(true);
				const Wheel *wheel = p.first;
				float newSize = rect.size().width / HOUGH_INITIAL_NUM_BINS;
				if (newSize > gCirclePrecision)
				{
					#ifdef _BENCHMARK_
						begin = std::chrono::high_resolution_clock::now();
					#endif 
					size_t binX = index % HOUGH_INITIAL_NUM_BINS;
					size_t binY = index / HOUGH_INITIAL_NUM_BINS;
					// calculate new rect 
					float x = rect.tl().x + binX * newSize;
					float y = rect.tl().y + binY * newSize;
					cv::Rect2f newRect(x, y, newSize, newSize);
					// create new sampler 
					Sampler newSampler(sampler);
					std::vector<size_t> points;
					for (size_t i = 0; i < component->numPoints(); i++)
					{
						if (!newSampler.isRemoved(i))
						{
							if (!pointIntersectsRect(component->point(i), newRect))
							{
								newSampler.removePoint(i);
							}
							else
							{
								points.push_back(i);
							}
						}
					}
					#ifdef _BENCHMARK_
						end = std::chrono::high_resolution_clock::now();
						benchmarkCreateComponent += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
					#endif 
					// call recursively
					findCircles(component, newSampler, newRect, circles, wheel);
					// remove points  
					for (const size_t &point : points)
					{
						if (newSampler.isRemoved(point))
						{
							sampler.removePoint(point);
						}
					}
				}
				else 
				{
					Circle *circle = wheel->getCircle();
					circles.push_back(circle);
					#ifdef _BENCHMARK_
						begin = std::chrono::high_resolution_clock::now();
					#endif 
					for (size_t i = 0; i < component->numPoints(); i++)
					{
						if (!sampler.isRemoved(i))
						{
							cv::Point2f diff = component->point(i)->position - circle->center;
							float dist = std::abs(std::sqrt(diff.x * diff.x + diff.y * diff.y) - circle->radius);
							if (dist < gCirclePrecision)
							{
								sampler.removePoint(i);
							}
						}
					}
					#ifdef _BENCHMARK_
						end = std::chrono::high_resolution_clock::now();
						benchmarkCreateComponent += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
					#endif 
				}
			}
		}
	}
	while (sampler.canSample())
	{
		std::pair<size_t, size_t> sample = sampler.sample();
		std::shared_ptr<Intersection> intersection = intersectionBetweenPoints(component->point(sample.first), component->point(sample.second));
		if (intersection != NULL)
		{
			sampler.removePoint(sample.first);
			sampler.removePoint(sample.second);
			if (rect.contains(intersection->position))
			{
				#ifdef _BENCHMARK_
					begin = std::chrono::high_resolution_clock::now();
				#endif 
				intersection->radiusBin = calculateRadiusBin(intersection->dist, component->rect().size().width);
				size_t binX, binY;
				calculateAccumulatorBin(intersection->position, rect, binX, binY);
				size_t index = binY * HOUGH_INITIAL_NUM_BINS + binX;
				#ifdef _BENCHMARK_
					end = std::chrono::high_resolution_clock::now();
					benchmarkIntersection += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
				#endif 
				if (accumulators[index].visited()) continue;
				const Wheel *wheel = accumulators[index].addIntersection(intersection);
				if (wheel != NULL)
				{
					accumulators[index].visited(true);
					float newSize = rect.size().width / HOUGH_INITIAL_NUM_BINS;
					// check if needed precision is achieved 
					if (newSize > gCirclePrecision)
					{
						#ifdef _BENCHMARK_
							begin = std::chrono::high_resolution_clock::now();
						#endif 
						// calculate new rect 
						float x = rect.tl().x + binX * newSize;
						float y = rect.tl().y + binY * newSize;
						cv::Rect2f newRect(x, y, newSize, newSize);
						// create new sampler 
						Sampler newSampler(sampler);
						std::vector<size_t> points;
						for (size_t i = 0; i < component->numPoints(); i++)
						{
							if (!newSampler.isRemoved(i))
							{
								if (!pointIntersectsRect(component->point(i), newRect))
								{
									newSampler.removePoint(i);
								}
								else
								{
									points.push_back(i);
								}
							}
						}
						#ifdef _BENCHMARK_
							end = std::chrono::high_resolution_clock::now();
							benchmarkCreateComponent += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
						#endif 
						// call recursively
						findCircles(component, newSampler, newRect, circles, wheel);
						// remove points 
						for (const size_t &point : points)
						{
							if (newSampler.isRemoved(point))
							{
								sampler.removePoint(point);
							}
						}
					}
					else
					{
						Circle *circle = wheel->getCircle();
						circles.push_back(circle);
						#ifdef _BENCHMARK_
							begin = std::chrono::high_resolution_clock::now();
						#endif 
						for (size_t i = 0; i < component->numPoints(); i++)
						{
							if (!sampler.isRemoved(i))
							{
								cv::Point2f diff = component->point(i)->position - circle->center;
								float dist = std::abs(std::sqrt(diff.x * diff.x + diff.y * diff.y) - circle->radius);
								if (dist < gCirclePrecision)
								{
									sampler.removePoint(i);
								}
							}
						}
						#ifdef _BENCHMARK_
							end = std::chrono::high_resolution_clock::now();
							benchmarkCreateComponent += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
						#endif 
					}
				}
			}
			else 
			{
				intersection.reset();
			}
		}
	}
	#ifdef _BENCHMARK_
		begin = std::chrono::high_resolution_clock::now();
	#endif 
	delete[] accumulators;
	#ifdef _BENCHMARK_
		end = std::chrono::high_resolution_clock::now();
		benchmarkInitialization += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
}

void calculateNormalsAndCurvatures(const cv::Mat &gray, const cv::Mat &edges, std::vector<Point*> &points)
{
	points.resize(gray.rows * gray.cols);
	cv::Mat gradX, gradY, laplacian;
	cv::Sobel(gray, gradX, CV_16S, 1, 0);
	cv::Sobel(gray, gradY, CV_16S, 0, 1);
	cv::Laplacian(gray, laplacian, CV_16S);
	const uchar *edgePtr = (uchar*)edges.data;
	const short *gradXPtr = (short*)gradX.data;
	const short *gradYPtr = (short*)gradY.data;
	const short *laplacianPtr = (short*)laplacian.data;
	int index = 0;
	for (int y = 0; y < gray.rows; y++)
	{
		for (int x = 0; x < gray.cols; x++)
		{
			if (edgePtr[index])
			{
				short gradX = gradXPtr[index];
				short gradY = gradYPtr[index];
				float norm = std::sqrt(static_cast<float>(gradX * gradX + gradY * gradY));
				if (norm > std::numeric_limits<float>::epsilon())
				{
					Point *point = new Point;
					point->position = cv::Point2f(x + 0.5f, y + 0.5f);
					point->normal = cv::Point2f(gradX / norm, gradY / norm);
					point->curvature = std::abs(*laplacianPtr);
					points[index] = point;
				}
				else 
				{
					points[index] = NULL;
				}
			}
			else 
			{
				points[index] = NULL;
			}
			++index;
		}
	}
}

Component* createNewComponent(Point *seed)
{
	Component *component = new Component;
	std::queue<Point*> queue;
	queue.push(seed);
	while (!queue.empty())
	{
		Point *current = queue.front();
		queue.pop();
		if (current != NULL && !current->marked)
		{
			current->marked = true;
			component->addPoint(current);
			for (int i = 0; i < 8; i++)
			{
				queue.push(current->neighbors[i]);
			}
		}
	}
	return component;
}

void findConnectedComponents(int width, int height, const std::vector<Point*> &points, std::vector<Component*> &components)
{
	// create neighborhood 
	int yTimesWidth[height];
	for (int y = 0; y < height; y++)
	{
		yTimesWidth[y] = y * width;
	}
	int index = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (points[index] != NULL)
			{
				int neighbor = -1;
				//8-neighborhood
				for (int y_ = -1; y_ < 2; y_++)
				{
					for (int x_ = -1; x_ < 2; x_++)
					{
						if (y_ == 0 && x_ == 0) continue;
						++neighbor;
						int neighborY = y + y_;
						int neighborX = x + x_;
						if (neighborY < 0 || neighborX < 0 || neighborY == height || neighborX == width) continue;
						int neighborIndex = yTimesWidth[neighborY] + neighborX;
						if (points[neighborIndex] != NULL &&
							((neighborIndex < index && points[neighborIndex]->neighbors[7 - neighbor] == points[index]) || 
							 (neighborIndex > index && std::abs(points[index]->normal.dot(points[neighborIndex]->normal) > COMPONENTS_GRADIENT_THRESHOLD))))
						{
							points[index]->neighbors[neighbor] = points[neighborIndex];
						}
					}
				}
			}
			++index;
		}
	}
	// create components 
	index = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (points[index] != NULL && !points[index]->marked)
			{
				Component *component = createNewComponent(points[index]);
				if (component->numPoints() < ACCUMULATOR_MIN_ARC_LENGTH)
				{
					delete component;
				}
				else 
				{
					component->calculateRect();
					component->reorientNormals();
					components.push_back(component);
				}
			}
			++index;
		}
	}
}

void cannyCallback(int slider, void *userData)
{
	std::cout << "Preprocessing..." << std::endl;
	auto begin = std::chrono::high_resolution_clock::now();
	cv::Mat gray = gImgGray.clone();
	// Canny detector 
	cv::Mat edges;
	cv::Canny(gray, edges, gCannyLowThreshold, gCannyLowThreshold * CANNY_RATIO, CANNY_KERNEL_SIZE);
	// Calculate normals and curvatures  
	std::vector<Point*> points;
	// Reduce noise 
	calculateNormalsAndCurvatures(gray, edges, points);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "Time elapsed: " << duration << "ms..." << std::endl;
	// Find connected components  
	std::cout << "Finding connected components..." << std::endl;
	begin = std::chrono::high_resolution_clock::now();
	std::vector<Component*> components;
	findConnectedComponents(gray.cols, gray.rows, points, components);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "Time elapsed: " << duration << "ms..." << std::endl;
	// Find circles 
	std::cout << "Detecting circles..." << std::endl;
	#ifdef _BENCHMARK_
		benchmarkInitialization = 0;
		benchmarkSample = 0;
		benchmarkIntersection = 0;
		benchmarkAddIntersection = 0;
		benchmarkCreateComponent = 0;
	#endif 
	begin = std::chrono::high_resolution_clock::now();
	std::vector<Circle*> circles;
	for (const Component *component : components)
	{
		std::vector<Circle*> detectedCircles;
		Sampler sampler(component);
		findCircles(component, sampler, component->rect(), detectedCircles);
		for (Circle *circle : detectedCircles)
		{
			circles.push_back(circle);
		}
	}
	/*std::vector<Circle*> uniqueCircles;
	for (size_t i = 0; i < circles.size(); i++)
	{
		bool isRepeated = false;
		for (size_t j = i + 1; j < circles.size(); j++)
		{
			if (std::abs(circles[i]->radius - circles[j]->radius) < gCirclePrecision)
			{
				cv::Point2f diff = circles[i]->center - circles[j]->center;
				float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);
				if (dist < gCirclePrecision)
				{
					isRepeated = true;
					delete circles[j];
					break;
				}
			}
		}
		if (!isRepeated)
		{
			uniqueCircles.push_back(circles[i]);
		}
	}
	circles = uniqueCircles;*/
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	#ifdef _BENCHMARK_
		std::cout << "Initialization time elapsed: " << benchmarkInitialization / 1e6 << "ms..." << std::endl;
		std::cout << "Sample time elapsed: " << benchmarkSample / 1e6 << "ms..." << std::endl;
		std::cout << "Intersection time elapsed: " << benchmarkIntersection / 1e6 << "ms..." << std::endl;
		std::cout << "Add intersection time elapsed: " << benchmarkAddIntersection / 1e6 << "ms..." << std::endl;
		std::cout << "Create component: " << benchmarkCreateComponent / 1e6 << "ms..." << std::endl;
	#endif 
	std::cout << "Total time elapsed: " << duration << "ms..." << std::endl;
	// Create component colors 
	std::map<const Component*, cv::Scalar> colors;
	for (const Component *component : components)
	{
		colors[component] = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
	}
	// Draw points 
	cv::Mat img = cv::Mat::zeros(edges.size(), CV_8UC3);
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			int index = y * img.cols + x;
			if (points[index] != NULL)
			{
				int x = static_cast<int>(points[index]->position.x - 0.5f);
				int y = static_cast<int>(points[index]->position.y - 0.5f);
				cv::Scalar color = cv::Scalar(255, 255, 255);
				img.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
			}
		}
	}
	// Draw normals 
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			int index = y * img.cols + x;
			if (points[index] != NULL)
			{
				int x = static_cast<int>(points[index]->position.x - 0.5f);
				int y = static_cast<int>(points[index]->position.y - 0.5f);
				int normalX = static_cast<int>(points[index]->normal.x * 10);
				int normalY = static_cast<int>(points[index]->normal.y * 10);
				//cv::line(img, cv::Point(x, y), cv::Point(x + normalX, y + normalY), cv::Scalar(0, 0, 255));
			}
		}
	}
	// Draw circles 
	for (const Circle *circle : circles)
	{
		cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
		cv::circle(img, cv::Point((int)circle->center.x, (int)circle->center.y), (int)circle->radius, color, 2); 
	}
	// Remove points
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		if (*it != NULL)
			delete *it;
	}
	// Display image
	cv::imshow(gEdgeWindowName, img);
}

int main(int argc, char **argv)
{
	std::srand(time(NULL));

	if (argc < 4)
	{
		std::cerr << "Usage: ARHT <INPUT_IMAGE> <CANNY_MIN_THRESHOLD> <CIRCLE_PRECISION>" << std::endl;
		return -1;
	}
	std::string inputFilename(argv[1]);
	gImg = cv::imread(inputFilename);
	cv::cvtColor(gImg, gImgGray, CV_BGR2GRAY);
	gCannyLowThreshold = atoi(argv[2]);
	gCirclePrecision = atoi(argv[3]);
	if (!gImg.data)
	{
		std::cerr << "ERROR: Could not read image." << std::endl;
		return -1;
	}
	cv::namedWindow(gEdgeWindowName, CV_WINDOW_AUTOSIZE);
	cv::createTrackbar("Min threshold:", gEdgeWindowName, &gCannyLowThreshold, CANNY_MAX_THRESHOLD, cannyCallback);
	cannyCallback(gCannyLowThreshold, NULL);
	cv::imshow("Original image", gImg);
	cv::waitKey(0);

	return 0;
}
