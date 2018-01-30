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

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>

#include "houghcell.h" 

std::string gEdgeWindowName = "Edge image";
int gCannyLowThreshold;
int gMinCellSize;
int gCellBranchingFactor;
int gNumAngles;
int gMinArcLength;
cv::Mat gImg;
cv::Mat gImgGray;
cv::Mat gFrame;

#ifdef _BENCHMARK
double gTimeProcessImage;
double gTimeCreateConnectedComponents;
double gTimeGroupPoints;
double gTimeCreatePointCloud;
double gTimeCreateSampler;
double gTimeSample1;
double gTimeSample2;
double gTimeIntersection;
double gTimeAddIntersection;
double gTimeBlockPoints;
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
std::vector<Circle> gCircles;
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

void drawCircle(const Circle &circle, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	cv::circle(gFrame, cv::Point((int)circle.center.x, (int)circle.center.y), (int)circle.radius, color, 2); 
}

void drawPoints(const PointCloud &pointCloud, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	for (size_t i = 0; i < pointCloud.numPoints(); i++)
	{
		drawPoint(pointCloud.point(i).position, color);
	}
}

void drawGroups(const PointCloud &pointCloud, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	for (size_t i = 0; i < pointCloud.numGroups(); i++)
	{
		drawPoint(pointCloud.group(i).position, color);
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
void displayInteractiveFrame(const std::vector<PointCloud> &pointClouds)
{
	cv::Mat img = gFrame.clone();
	for (const cv::Rect2f &rect : gRects)
	{
		drawRect(rect, cv::Scalar(0, 0, 255));
	}
	drawRect(gActiveCell->extension(), cv::Scalar(0, 255, 0));
	for (const PointCloud &pointCloud : pointClouds)
	{
		Sampler *sampler = pointCloud.sampler();
		for (size_t i = 0; i < sampler->numPoints(); i++)
		{
			if (sampler->isRemoved(i))
			{
				drawPoint(pointCloud.group(i).position, cv::Scalar(0, 0, 255));
			}
			else if (sampler->isBlocked(i))
			{
				drawPoint(pointCloud.group(i).position, cv::Scalar(150, 150, 150));
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
		const Point &p1 = intersection.sampler->pointCloud().group(intersection.p1);
		const Point &p2 = intersection.sampler->pointCloud().group(intersection.p2);
		drawLine(p1.position, p1.position + p1.normal * 10, color);
		drawLine(p2.position, p2.position + p2.normal * 10, color);
		drawPoint(intersection.position, color);
	}
	for (const Circle &circle : gCircles)
	{
		drawCircle(circle, cv::Scalar(255, 0, 0));
	}
	cv::imshow(gEdgeWindowName, gFrame);
	cv::waitKey(0);
	gFrame = img;
}
#endif 

bool isPointOnCircle(const HoughCell *cell, const Circle &circle, const cv::Point2f &point)
{
	float dist = norm(point - circle.center);
	return std::abs(dist - circle.radius) < cell->minCellSize();
}

bool circleIntersectsRect(const Circle &circle, const cv::Rect2f &rect)
{
	cv::Rect2f boundingBox(circle.center.x - circle.radius, circle.center.y - circle.radius, circle.radius * 2, circle.radius * 2);
	cv::Point2f boundingBoxCorners[4] = {
		cv::Point2f(boundingBox.x, boundingBox.y),
		cv::Point2f(boundingBox.x, boundingBox.y + boundingBox.height),
		cv::Point2f(boundingBox.x + boundingBox.width, boundingBox.y),
		cv::Point2f(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height)
	};
	cv::Point2f rectCorners[4] = {
		cv::Point2f(rect.x, rect.y),
		cv::Point2f(rect.x, rect.y + rect.height),
		cv::Point2f(rect.x + rect.width, rect.y),
		cv::Point2f(rect.x + rect.width, rect.y + rect.height)
	};
	for (size_t i = 0; i < 4; i++)
	{
		if (rect.contains(boundingBoxCorners[i]) || boundingBox.contains(rectCorners[i]))
			return true;
	}
	return false;
}

void removeCirclePoints(const HoughCell *cell, const Circle &circle, const std::vector<PointCloud> &pointClouds)
{
	for (const PointCloud &pointCloud : pointClouds)
	{
		if (circleIntersectsRect(circle, pointCloud.extension()))
		{
			Sampler *sampler = pointCloud.sampler();
			for (size_t i = 0; i < sampler->numPoints(); i++)
			{
				if (!sampler->isRemoved(i) && isPointOnCircle(cell, circle, pointCloud.group(i).position))
				{
					sampler->removePoint(i);
				}
			}
		}
	}
}

void houghTransform(HoughCell *cell, const std::vector<PointCloud> &pointClouds, std::vector<Circle> &circles);

void subdivide(HoughAccumulator *accumulator, const std::vector<PointCloud> &pointClouds, std::vector<Circle> &circles)
{
	HoughCell *cell = accumulator->cell();
	#ifdef _DEBUG_INTERACTIVE
		gRects.push_back(cell->extension());
	#endif
	if (cell->cellSize() > cell->minCellSize())
	{
		for (HoughAccumulator *childAccumulator : cell->visit(pointClouds))
		{
			if (childAccumulator->hasCircleCandidate())
			{
				subdivide(childAccumulator, pointClouds, circles);
			} 
		}
		houghTransform(cell, pointClouds, circles);
	}        
	else
	{
		#ifdef _BENCHMARK
			auto begin = std::chrono::high_resolution_clock::now();
		#endif
		Circle circle = accumulator->getCircleCandidate();
		circles.push_back(circle);
		cell->setVisited();
		removeCirclePoints(cell, circle, pointClouds);
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeAddCircle += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		#endif 
		#ifdef _DEBUG_INTERACTIVE
			gActiveCell = cell;
			gCircles.push_back(circle);
			displayInteractiveFrame(pointClouds);
		#endif
	}
}
 
void houghTransform(HoughCell *cell, const std::vector<PointCloud> &pointClouds, std::vector<Circle> &circles)
{
	cell->blockPoints(pointClouds);
	for (const PointCloud &pointCloud : pointClouds)
	{
		Sampler *sampler = pointCloud.sampler();
		size_t numSamples = 0;
		while (sampler->canSample() && ++numSamples < sampler->numPoints())
		{
			HoughAccumulator *accumulator = cell->addIntersection(sampler);
			if (accumulator != NULL)
			{
				#ifdef _DEBUG_INTERACTIVE
					gActiveCell = cell;
					gIntersections.push_back(accumulator->intersections()[accumulator->intersections().size() - 1]);
					displayInteractiveFrame(pointClouds);
				#endif
				if (accumulator->hasCircleCandidate())
				{
					cell->unblockPoints(pointClouds);
					subdivide(accumulator, pointClouds, circles);
					cell->blockPoints(pointClouds);
				}
			}
		}
	}
	cell->unblockPoints(pointClouds);
} 
 
void cannyCallback(int slider, void *userData)
{
	#ifdef _BENCHMARK
		gTimeProcessImage = 0;
		gTimeCreateConnectedComponents = 0;
		gTimeGroupPoints = 0;
		gTimeCreatePointCloud = 0;
		gTimeCreateSampler = 0;
		gTimeSample1 = 0;
		gTimeSample2 = 0;
		gTimeIntersection = 0;
		gTimeAddIntersection = 0;
		gTimeBlockPoints = 0;
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
	std::cout << "Image size: " << gray.size() << std::endl;
	// Calculate normals and curvatures 
	std::vector<PointCloud> pointClouds;
	cv::Rect2f extension = PointCloud::createPointCloudsFromImage(gray, gCannyLowThreshold, gNumAngles, pointClouds);
	std::cout << "Extension: " << extension << std::endl;
	for (PointCloud &pointCloud : pointClouds)
	{
		pointCloud.createSampler(gMinArcLength);
	}
	HoughCell *cell = new HoughCell(extension, gMinArcLength, gMinCellSize, gCellBranchingFactor);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "Time elapsed: " << duration / 1e6 << "ms" << std::endl;
	#ifdef _BENCHMARK
		std::cout << "\tTime to process image: " << gTimeProcessImage / 1e6 << "ms (" << gTimeProcessImage / duration * 100 << "%)" << std::endl
					<< "\tTime to create connected components: " << gTimeCreateConnectedComponents / 1e6 << "ms (" << gTimeCreateConnectedComponents / duration * 100 << "%)" << std::endl
					<< "\tTime to group points: " << gTimeGroupPoints / 1e6 << "ms (" << gTimeGroupPoints / duration * 100 << "%)" << std::endl
					<< "\tTime to create point cloud: " << gTimeCreatePointCloud / 1e6 << "ms (" << gTimeCreatePointCloud / duration * 100 << "%)" << std::endl
					<< "\tTime to create sampler: " << gTimeCreateSampler / 1e6 << "ms (" << gTimeCreateSampler / duration * 100 << "%)" << std::endl
					<< "\tExplained: " << (gTimeProcessImage + gTimeCreateConnectedComponents + gTimeGroupPoints + gTimeCreatePointCloud + gTimeCreateSampler) / duration * 100 << "%" << std::endl;
	#endif 
	// Draw points 
	gFrame = cv::Mat::zeros(gray.size(), CV_8UC3);
	#ifdef _DEBUG_INTERACTIVE	
		for (size_t i = 0; i < pointClouds.size(); i++)
		{
			drawPoints(pointClouds[i]);
		}
		cv::imshow(gEdgeWindowName, gFrame);
		cv::waitKey(0);
		std::vector<cv::Scalar> colors;
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		for (size_t i = 0; i < pointClouds.size(); i++)
		{
			color = cv::Scalar(((int)color[0] + 50 + rand() % 150) % 255, ((int)color[1] + 50 + rand() % 150) % 255, ((int)color[2] + 50 + rand() % 150) % 255);
			colors.push_back(color);
		}
		for (size_t i = 0; i < pointClouds.size(); i++)
		{
			drawPoints(pointClouds[i], colors[i]);
		}
		cv::imshow(gEdgeWindowName, gFrame);
		cv::waitKey(0);
		gFrame = cv::Mat::zeros(gray.size(), CV_8UC3);
		for (size_t i = 0; i < pointClouds.size(); i++)
		{
			drawGroups(pointClouds[i], colors[i]);
		}
		cv::imshow(gEdgeWindowName, gFrame);
		cv::waitKey(0);
		for (PointCloud &pointCloud : pointClouds)
		{
			drawQuadtree(&pointCloud.sampler()->quadtree());
		}
		cv::imshow(gEdgeWindowName, gFrame);
		cv::waitKey(0);
		gFrame = cv::Mat::zeros(gray.size(), CV_8UC3);
		for (size_t i = 0; i < pointClouds.size(); i++)
		{
			drawGroups(pointClouds[i], colors[i]);
		}
	#endif 
	#ifndef _DEBUG_INTERACTIVE
		for (size_t i = 0; i < pointClouds.size(); i++)
		{
			drawPoints(pointClouds[i]);
		}
	#endif 	
	// Find circles 
	std::cout << "Detecting circles..." << std::endl;
	begin = std::chrono::high_resolution_clock::now();
	std::vector<Circle> circles;
	#ifdef _DEBUG_INTERACTIVE
		gActiveCell = cell;
		gRects.push_back(cell->extension());
	#endif
	houghTransform(cell, pointClouds, circles);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "Time elapsed: " << duration / 1e6 << "ms" << std::endl;
	#ifdef _BENCHMARK
		std::cout << "\tTime to sample: " << (gTimeSample1 + gTimeSample2) / 1e6 << "ms (" << (gTimeSample1 + gTimeSample2) / duration * 100 << "%)" << std::endl
						<< "\t\tFirst point: " << gTimeSample1 / 1e6 << "ms (" << gTimeSample1 / duration * 100 << "%); Second point: " << gTimeSample2 / 1e6 << "ms (" << gTimeSample2 / duration * 100 << "%)" << std::endl
					<< "\tTime to calculate intersection: " << gTimeIntersection / 1e6 << "ms (" << gTimeIntersection / duration * 100 << "%)" << std::endl
					<< "\tTime to add intersection: " << gTimeAddIntersection / 1e6 << "ms (" << gTimeAddIntersection / duration * 100 << "%)" << std::endl
					<< "\tTime to block points: " << gTimeBlockPoints / 1e6 << "ms (" << gTimeBlockPoints / duration * 100 << "%)" << std::endl
					<< "\tTime to visit cell: " << gTimeVisit / 1e6 << "ms (" << gTimeVisit / duration * 100 << "%)" << std::endl
					<< "\tTime to add circle: " << gTimeAddCircle / 1e6 << "ms (" << gTimeAddCircle / duration * 100 << "%)" << std::endl
					<< "\tTime debug: " << gTimeDebug / 1e6 << "ms (" << gTimeDebug / duration * 100 << "%)" << std::endl
					<< "\tExplained: " << (gTimeSample1 + gTimeSample2 + gTimeIntersection + gTimeAddIntersection + gTimeBlockPoints + gTimeVisit + gTimeAddCircle) / duration * 100 << "%" << std::endl;
	#endif 
	// Draw circles 
	for (const Circle &circle : circles)
	{
		drawCircle(circle, cv::Scalar(rand() % 255, rand() % 255, rand() % 255));
	}
	// Delete data
	delete cell;
	// Display image
	cv::imshow(gEdgeWindowName, gFrame);
}

int main(int argc, char **argv)
{
	std::srand(time(NULL));

	if (argc < 5)
	{
		std::cerr << "Usage: ARHT <INPUT_IMAGE> <CANNY_MIN_THRESHOLD> <MIN_CELL_SIZE> <CELL_BRANCHING_FACTOR> <NUM_ANGLES> <MIN_ARC_LENGTH>" << std::endl;
		return -1;
	}
	std::string inputFilename(argv[1]);
	gImg = cv::imread(inputFilename);
	cv::cvtColor(gImg, gImgGray, CV_BGR2GRAY);
	gCannyLowThreshold = atoi(argv[2]);
	gMinCellSize = atoi(argv[3]);
	gCellBranchingFactor = atoi(argv[4]);
	gNumAngles = atoi(argv[5]);
	gMinArcLength = atoi(argv[6]);
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
