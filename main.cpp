//#define _BENCHMARK_
//#define _DEBUG_INTERACTIVE

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

#include "houghcell.h"

// pre-processing parameters
#define CANNY_MAX_THRESHOLD 100
#define CANNY_RATIO 3
#define CANNY_KERNEL_SIZE 3

// algorithm parameters 
#define HOUGH_NUM_ANGLES 18
#define HOUGH_MIN_ARC_LENGTH 9 
#define HOUGH_BRANCHING_FACTOR 2 
#define HOUGH_MAX_INTERSECTION_RATIO 1.5f
#define QUADTREE_MIN_NUM_POINTS 25 
#define SAMPLER_CLIMB_CHANCE 0.5f 

std::string gEdgeWindowName = "Edge image";
int gCannyLowThreshold;
int gCirclePrecision;
cv::Mat gImg;
cv::Mat gImgGray;
cv::Mat gFrame;

#ifdef _DEBUG_INTERACTIVE
struct Intersection;
struct Circle;
struct HoughCell;
std::vector<cv::Rect2f> gRects;
std::vector<Intersection> gIntersections;
std::vector<Circle*> gCircles;
HoughCell *gActiveCell;
#endif 

void drawPoint(const cv::Point2f &position, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	int x = (int)position.x;
	int y = (int)position.y;
	if (x >= 0 && y >= 0 && x < gFrame.cols && y < gFrame.rows)
		gFrame.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
}

void drawLine(const cv::Point2f &p1, const cv::Point2f &p2, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	int x1 = (int)p1.x;
	int y1 = (int)p1.y;
	int x2 = (int)p2.x;
	int y2 = (int)p2.y;
	cv::line(gFrame, cv::Point2i(x1, y1), cv::Point2i(x2, y2), color);
}

void drawRect(const cv::Rect2f &rect, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	cv::Rect2i r((int)rect.x, (int)rect.y, (int)rect.width, (int)rect.height);
	cv::rectangle(gFrame, r, color);
}

void drawCircle(const Circle *circle, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	cv::circle(gFrame, cv::Point((int)circle->center.x, (int)circle->center.y), (int)circle->radius, color, 2); 
}

void drawPoints(const PointCloud *pointCloud)
{
	for (size_t i = 0; i < pointCloud->numPoints(); i++)
	{
		drawPoint(pointCloud->point(i)->position);
	}
}
  
void drawQuadtree(const Quadtree *quadtree)
{
	drawRect(quadtree->rect(), cv::Scalar(0, 255, 0));
	for (size_t i = 0; i < 4; i++)
	{
		if (quadtree->child(i) != NULL)   
		{ 
			drawQuadtree(quadtree->child(i));  
		}
	}
}


#ifdef _DEBUG_INTERACTIVE
void displayInteractiveFrame()
{
	cv::Mat img = gFrame.clone();
	const PointCloud *pointCloud = gActiveCell->pointCloud();
	for (const cv::Rect2f &rect : gRects)
	{
		drawRect(rect, cv::Scalar(0, 0, 255));
	}
	drawRect(gActiveCell->rect(), cv::Scalar(0, 255, 0));
	if (gActiveCell->sampler() != NULL)
	{
		for (size_t i = 0; i < pointCloud->numPoints(); i++)
		{
			if (gActiveCell->sampler()->isRemoved(i))
			{
				drawPoint(pointCloud->point(i)->position, cv::Scalar(0, 0, 255));
			}
		}
	}
	for (const Intersection &intersection : gIntersections)
	{
		cv::Scalar color;
		if (intersection.position == gIntersections[gIntersections.size() - 1].position)
			color = cv::Scalar(0, 255, 0);
		else 
			color = cv::Scalar(0, 0, 255);
		const Point *p1 = pointCloud->point(intersection.p1);
		const Point *p2 = pointCloud->point(intersection.p2);
		drawLine(p1->position, p1->position + p1->normal * 10, color);
		drawLine(p2->position, p2->position + p2->normal * 10, color);
		drawPoint(intersection.position, color);
	}
	for (const Circle *circle : gCircles)
	{
		drawCircle(circle, cv::Scalar(255, 0, 0));
	}
	cv::imshow(gEdgeWindowName, gFrame);
	cv::waitKey(0);
	gFrame = img;
}
#endif 


bool isPointOnCircle(HoughCell *cell, Circle *circle, size_t point)
{
	cv::Point2f pointPosition = cell->pointCloud()->point(point)->position;
	float dist = norm(pointPosition - circle->center);
	return std::abs(dist - circle->radius) < cell->minCellSize();
}

void houghTransform(HoughCell *cell, std::vector<Circle*> &circles);

void subdivide(HoughAccumulator *accumulator, std::vector<Circle*> &circles)
{
	HoughCell *cell = accumulator->cell();
	#ifdef _DEBUG_INTERACTIVE
		gRects.push_back(cell->rect());
	#endif
	if (cell->cellSize() > cell->minCellSize())
	{
		for (HoughAccumulator *childAccumulator : cell->visit())
		{
			if (childAccumulator->hasCircleCandidate())
			{
				subdivide(childAccumulator, circles);
			} 
		}
		houghTransform(cell, circles);
	}        
	else
	{
		Circle *circle = accumulator->getCircleCandidate();
		for (size_t i = 0; i < cell->pointCloud()->numPoints(); i++)
		{
			if (!cell->parent()->sampler()->isRemoved(i) && isPointOnCircle(cell, circle, i))
			{
				cell->parent()->sampler()->removePointFromAll(i);
			}
		} 
		#ifdef _DEBUG_INTERACTIVE
			gActiveCell = cell;
			gCircles.push_back(circle);
			displayInteractiveFrame();
		#endif
		circles.push_back(circle);
		cell->setVisited();
	}
}
 
void houghTransform(HoughCell *cell, std::vector<Circle*> &circles)
{
	size_t numSamples = 0;
	while (!cell->isTermined() && numSamples < cell->sampler()->numAvailablePoints())
	{
		HoughAccumulator *accumulator = cell->addIntersection();
		if (accumulator != NULL) 
		{
			#ifdef _DEBUG_INTERACTIVE
				gActiveCell = cell;
				gIntersections.push_back(accumulator->intersections()[accumulator->intersections().size() - 1]);
				displayInteractiveFrame();
			#endif
			if (accumulator->hasCircleCandidate())
			{
				subdivide(accumulator, circles);
			}
		}   
		++numSamples;      
	}
} 
 
void cannyCallback(int slider, void *userData)
{
	#ifdef _DEBUG_INTERACTIVE	
		gRects.clear();
		gIntersections.clear();
		gCircles.clear();
	#endif 
	std::cout << "Preprocessing..." << std::endl;
	auto begin = std::chrono::high_resolution_clock::now();
	cv::Mat gray = gImgGray.clone();
	// Canny detector 
	cv::Mat edges;
	cv::Canny(gray, edges, gCannyLowThreshold, gCannyLowThreshold * CANNY_RATIO, CANNY_KERNEL_SIZE);
	// Calculate normals and curvatures  
	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0);
	std::cout << "a" << std::endl;
	PointCloud *pointCloud = new PointCloud(gray, edges, HOUGH_NUM_ANGLES);
	std::cout << "b" << std::endl;
	Quadtree *quadtree = new Quadtree(pointCloud, QUADTREE_MIN_NUM_POINTS);
	std::cout << "c" << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "Time elapsed: " << duration << "ms..." << std::endl;
	// Draw points 
	gFrame = cv::Mat::zeros(edges.size(), CV_8UC3);
	drawPoints(pointCloud);
	//drawQuadtree(quadtree);
	// Find circles 
	std::cout << "Detecting circles..." << std::endl;
	#ifdef _BENCHMARK_
 
	#endif 
	begin = std::chrono::high_resolution_clock::now();
	Sampler *sampler = new Sampler(quadtree, SAMPLER_CLIMB_CHANCE, HOUGH_MIN_ARC_LENGTH);
	HoughCell *cell = new HoughCell(sampler, HOUGH_BRANCHING_FACTOR, gCirclePrecision, HOUGH_MAX_INTERSECTION_RATIO);
	std::vector<Circle*> circles;
	#ifdef _DEBUG_INTERACTIVE
		gActiveCell = cell;
		gRects.push_back(cell->rect());
	#endif
	houghTransform(cell, circles);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	#ifdef _BENCHMARK_
		/*std::cout << "Initialization time elapsed: " << benchmarkInitialization / 1e6 << "ms..." << std::endl;
		std::cout << "Sample time elapsed: " << benchmarkSample / 1e6 << "ms..." << std::endl;
		std::cout << "Intersection time elapsed: " << benchmarkIntersection / 1e6 << "ms..." << std::endl;
		std::cout << "Create new bin: " << benchmarkCreateBin / 1e6 << "ms..." << std::endl;
		std::cout << "Add intersection time elapsed: " << benchmarkAddIntersection / 1e6 << "ms..." << std::endl;
		std::cout << "Call recursively: " << benchmarkRecursiveCall / 1e6 << "ms..." << std::endl;*/
	#endif 
	std::cout << "Total time elapsed: " << duration << "ms..." << std::endl;
	// Draw circles 
	for (const Circle *circle : circles)
	{
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		drawCircle(circle, color);
	}
	// Delete data
	//delete cell;
	delete quadtree;
	delete pointCloud;
	for (Circle *circle : circles)
	{
		delete circle;
	}
	// Display image
	cv::imshow(gEdgeWindowName, gFrame);
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
