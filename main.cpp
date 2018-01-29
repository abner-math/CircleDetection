#define _DEBUG_INTERACTIVE

#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <list>
#include <set>
#include <chrono>
#include <memory>
#include <time.h>

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>

#include "houghcell.h" 

// algorithm parameters 
#define HOUGH_NUM_ANGLES 18
#define HOUGH_MIN_ARC_LENGTH 9 
#define HOUGH_BRANCHING_FACTOR 2 
#define HOUGH_MAX_INTERSECTION_RATIO 1.5f
#define QUADTREE_MIN_NUM_POINTS 10
#define QUADTREE_MIN_NUM_ANGLES 3
#define QUADTREE_MIN_SIZE 20.0f
#define SAMPLER_CLIMB_CHANCE 0.25f 

std::string gEdgeWindowName = "Edge image";
int gCannyLowThreshold;
int gCirclePrecision;
cv::Mat gImg;
cv::Mat gImgGray;
cv::Mat gFrame;

#ifdef _BENCHMARK
double gTimeProcessImage;
double gTimeCreateConnectedComponents;
double gTimeCreatePointCloud;
double gTimeCreateQuadtree;
double gTimeSample1;
double gTimeSample2;
double gTimeIntersection;
double gTimeAddIntersection;
double gTimeVisit;
double gTimeAddCircle;
double gTimeDebug;
#endif 

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

void drawPoints(const PointCloud *pointCloud, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	for (size_t i = 0; i < pointCloud->numPoints(); i++)
	{
		drawPoint(pointCloud->point(i).position, color);
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
				drawPoint(pointCloud->point(i).position, cv::Scalar(0, 0, 255));
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
		const Point &p1 = pointCloud->point(intersection.p1);
		const Point &p2 = pointCloud->point(intersection.p2);
		drawLine(p1.position, p1.position + p1.normal * 10, color);
		drawLine(p2.position, p2.position + p2.normal * 10, color);
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
	const cv::Point2f &pointPosition = cell->pointCloud()->point(point).position;
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
		#ifdef _BENCHMARK
			auto begin = std::chrono::high_resolution_clock::now();
		#endif
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
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeAddCircle += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
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
	#ifdef _BENCHMARK
		gTimeProcessImage = 0;
		gTimeCreateConnectedComponents = 0;
		gTimeCreatePointCloud = 0;
		gTimeCreateQuadtree = 0;
		gTimeSample1 = 0;
		gTimeSample2 = 0;
		gTimeIntersection = 0;
		gTimeAddIntersection = 0;
		gTimeVisit = 0;
		gTimeAddCircle = 0;
		gTimeDebug = 0;
	#endif 
	#ifdef _DEBUG_INTERACTIVE	
		gRects.clear();
		gIntersections.clear();
		gCircles.clear();
	#endif 
	std::cout << "Preprocessing..." << std::endl;
	auto begin = std::chrono::high_resolution_clock::now();
	cv::Mat gray = gImgGray.clone();
	// Calculate normals and curvatures 
	std::vector<PointCloud*> pointClouds = PointCloud::createPointCloudsFromImage(gray, gCannyLowThreshold, HOUGH_NUM_ANGLES);
	PointCloud *pointCloud = pointClouds[0];
	/*Quadtree *quadtree = new Quadtree(pointCloud, QUADTREE_MIN_NUM_POINTS, QUADTREE_MIN_NUM_ANGLES, QUADTREE_MIN_SIZE);
	Sampler *sampler = new Sampler(quadtree, SAMPLER_CLIMB_CHANCE, HOUGH_MIN_ARC_LENGTH);
	HoughCell *cell = new HoughCell(sampler, HOUGH_BRANCHING_FACTOR, gCirclePrecision, HOUGH_MAX_INTERSECTION_RATIO);*/
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "Time elapsed: " << duration / 1e6 << "ms" << std::endl;
	#ifdef _BENCHMARK
		std::cout << "\tTime to process image: " << gTimeProcessImage / 1e6 << "ms (" << gTimeProcessImage / duration * 100 << "%)" << std::endl
					<< "\tTime to create connected components: " << gTimeCreateConnectedComponents / 1e6 << "ms (" << gTimeCreateConnectedComponents / duration * 100 << "%)" << std::endl
					<< "\tTime to create point cloud: " << gTimeCreatePointCloud / 1e6 << "ms (" << gTimeCreatePointCloud / duration * 100 << "%)" << std::endl
					<< "\tTime to create quadtree: " << gTimeCreateQuadtree / 1e6 << "ms (" << gTimeCreateQuadtree / duration * 100 << "%)" << std::endl
					<< "\tExplained: " << (gTimeProcessImage + gTimeCreateConnectedComponents + gTimeCreatePointCloud + gTimeCreateQuadtree) / duration * 100 << "%" << std::endl;
	#endif 
	// Draw points 
	gFrame = cv::Mat::zeros(gray.size(), CV_8UC3);
	for (size_t i = 0; i < pointClouds.size(); i++)
	{
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		drawPoints(pointClouds[i], color);
	}
	//drawQuadtree(quadtree);
	// Find circles 
	std::cout << "Detecting circles..." << std::endl;
	begin = std::chrono::high_resolution_clock::now();
	std::vector<Circle*> circles;
	#ifdef _DEBUG_INTERACTIVE
		//gActiveCell = cell;
		//gRects.push_back(cell->rect());
	#endif
	//houghTransform(cell, circles);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "Time elapsed: " << duration / 1e6 << "ms" << std::endl;
	#ifdef _BENCHMARK
		std::cout << "\tTime to sample: " << (gTimeSample1 + gTimeSample2) / 1e6 << "ms (" << (gTimeSample1 + gTimeSample2) / duration * 100 << "%)" << std::endl
						<< "\t\tFirst point: " << gTimeSample1 / 1e6 << "ms (" << gTimeSample1 / duration * 100 << "%); Second point: " << gTimeSample2 / 1e6 << "ms (" << gTimeSample2 / duration * 100 << "%)" << std::endl
					<< "\tTime to calculate intersection: " << gTimeIntersection / 1e6 << "ms (" << gTimeIntersection / duration * 100 << "%)" << std::endl
					<< "\tTime to add intersection: " << gTimeAddIntersection / 1e6 << "ms (" << gTimeAddIntersection / duration * 100 << "%)" << std::endl
					<< "\tTime to visit cell: " << gTimeVisit / 1e6 << "ms (" << gTimeVisit / duration * 100 << "%)" << std::endl
					<< "\tTime to add circle: " << gTimeAddCircle / 1e6 << "ms (" << gTimeAddCircle / duration * 100 << "%)" << std::endl
					<< "\tTime debug: " << gTimeDebug / 1e6 << "ms (" << gTimeDebug / duration * 100 << "%)" << std::endl
					<< "\tExplained: " << (gTimeSample1 + gTimeSample2 + gTimeIntersection + gTimeAddIntersection + gTimeVisit + gTimeAddCircle) / duration * 100 << "%" << std::endl;
	#endif 
	// Draw circles 
	cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
	for (const Circle *circle : circles)
	{
		color = cv::Scalar(((int)color[0] + 50 + rand() % 150) % 255, ((int)color[1] + 50 + rand() % 150) % 255, ((int)color[2] + 50 + rand() % 150) % 255);
		drawCircle(circle, color);
	}
	// Delete data
	//delete cell;
	//delete quadtree;
	for (Circle *circle : circles)
	{
		delete circle;
	}
	for (size_t i = 0; i < pointClouds.size(); i++)
	{
		delete pointClouds[i];
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
	cv::createTrackbar("Min threshold:", gEdgeWindowName, &gCannyLowThreshold, 100, cannyCallback);
	cannyCallback(gCannyLowThreshold, NULL);
	cv::imshow("Original image", gImg);
	cv::waitKey(0);

	return 0;
}
