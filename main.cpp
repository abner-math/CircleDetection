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

#include "houghcell.h" 

std::string gEdgeWindowName = "Edge image";
int gCannyLowThreshold;
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
double gTimeVisit;
double gTimeAddCircle;
double gTimeRemoveFalsePositive;
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

void drawCircle(const Circle &circle, cv::Scalar color = cv::Scalar(255, 255, 255), int thickness = 2)
{
	cv::circle(gFrame, cv::Point((int)circle.center.x, (int)circle.center.y), (int)circle.radius, color, thickness); 
}

void drawPoints(const PointCloud &pointCloud, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	for (size_t i = 0; i < pointCloud.numPoints(); i++)
	{
		drawPoint(pointCloud.point(i).position, color);
	}
}

void drawNormals(const PointCloud &pointCloud, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	for (size_t i = 0; i < pointCloud.numPoints(); i++)
	{
		drawLine(pointCloud.point(i).position, pointCloud.point(i).position + pointCloud.point(i).normal * 10, color);
	}
	drawPoint(pointCloud.center(), cv::Scalar(0, 255, 0));
}

void drawGroups(const PointCloud &pointCloud, cv::Scalar color = cv::Scalar(255, 255, 255))
{
	for (size_t i = 0; i < pointCloud.numGroups(); i++)
	{
		drawPoint(pointCloud.group(i).position, color);
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
	return std::abs(dist - circle.radius) < std::max(5.0f, circle.radius / 10);
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

void subdivide(HoughAccumulator *accumulator, const std::vector<PointCloud> &pointClouds, std::vector<Circle> &circles, float radius)
{
	HoughCell *cell = accumulator->cell();
	#ifdef _DEBUG_INTERACTIVE
		gActiveCell = cell;
		gRects.push_back(cell->extension());
	#endif
	if (cell->cellSize() > radius)
	{
		for (HoughAccumulator *childAccumulator : cell->visit())
		{
			if (childAccumulator->hasCircleCandidate())
			{
				subdivide(childAccumulator, pointClouds, circles, childAccumulator->radius());
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
			gCircles.push_back(circle);
			displayInteractiveFrame(pointClouds);
		#endif
	}
}
 
void houghTransform(HoughCell *cell, const std::vector<PointCloud> &pointClouds, std::vector<Circle> &circles)
{
	for (const PointCloud &pointCloud : pointClouds)
	{
		Sampler *sampler = pointCloud.sampler();
		size_t numSamples = 0;
		while (sampler->canSample())
		{
			HoughAccumulator *accumulator = cell->addIntersection(sampler);
			if (accumulator != NULL)
			{
				#ifdef _DEBUG_INTERACTIVE
					gActiveCell = cell;
					gIntersections.push_back(accumulator->intersections()[accumulator->intersections().size() - 1]);
					displayInteractiveFrame(pointClouds);
				#endif
				float radius;
				if (accumulator->hasCircleCandidate())
				{
					subdivide(accumulator, pointClouds, circles, accumulator->radius());
				}
			}
			++numSamples;
		}
	}
} 

void removeFalsePositiveCircles(std::vector<Circle> &circles)
{
	#ifdef _BENCHMARK
		auto begin = std::chrono::high_resolution_clock::now();
	#endif
	for (int i = circles.size() - 1; i >= 0; i--)
	{
		cv::Rect2i boundingBox((int)(circles[i].center.x - circles[i].radius), (int)(circles[i].center.y - circles[i].radius), (int)(circles[i].radius * 2), (int)(circles[i].radius * 2));
		cv::Mat edgeImg = PointCloud::edgeImg()(boundingBox);
		cv::Mat referenceImg = cv::Mat::zeros(cv::Size(boundingBox.width, boundingBox.height), CV_8U);
		cv::circle(referenceImg, cv::Point(boundingBox.width / 2, boundingBox.height / 2), (int)circles[i].radius, cv::Scalar(255, 255, 255), 2); 
		cv::Mat resultImg = referenceImg & edgeImg;
		float value = cv::sum(resultImg)[0] / 255.0f / (2 * M_PI * circles[i].radius);
		circles[i].removed = value < 0.25f;
	}
	#ifdef _BENCHMARK
		auto end = std::chrono::high_resolution_clock::now();
		gTimeRemoveFalsePositive += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	#endif 
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
		gTimeVisit = 0;
		gTimeAddCircle = 0;
		gTimeRemoveFalsePositive = 0;
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
	HoughCell *cell = new HoughCell(extension, gMinArcLength, gCellBranchingFactor);
	auto end = std::chrono::high_resolution_clock::now();
	auto durationPreprocessing = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "Time elapsed: " << durationPreprocessing / 1e6 << "ms" << std::endl;
	#ifdef _BENCHMARK
		std::cout << "\tTime to process image: " << gTimeProcessImage / 1e6 << "ms (" << gTimeProcessImage / durationPreprocessing * 100 << "%)" << std::endl
					<< "\tTime to create connected components: " << gTimeCreateConnectedComponents / 1e6 << "ms (" << gTimeCreateConnectedComponents / durationPreprocessing * 100 << "%)" << std::endl
					<< "\tTime to group points: " << gTimeGroupPoints / 1e6 << "ms (" << gTimeGroupPoints / durationPreprocessing * 100 << "%)" << std::endl
					<< "\tTime to create point cloud: " << gTimeCreatePointCloud / 1e6 << "ms (" << gTimeCreatePointCloud / durationPreprocessing * 100 << "%)" << std::endl
					<< "\tTime to create sampler: " << gTimeCreateSampler / 1e6 << "ms (" << gTimeCreateSampler / durationPreprocessing * 100 << "%)" << std::endl
					<< "\tExplained: " << (gTimeProcessImage + gTimeCreateConnectedComponents + gTimeGroupPoints + gTimeCreatePointCloud + gTimeCreateSampler) / durationPreprocessing * 100 << "%" << std::endl;
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
		gFrame = cv::Mat::zeros(gray.size(), CV_8UC3);
		for (size_t i = 0; i < pointClouds.size(); i++)
		{
			drawNormals(pointClouds[i], cv::Scalar(0, 0, 255));
		}
		cv::imshow(gEdgeWindowName, gFrame);
		cv::waitKey(0);
		gFrame = cv::Mat::zeros(gray.size(), CV_8UC3);
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
		gFrame = cv::Mat::zeros(gray.size(), CV_8UC3);
		for (size_t i = 0; i < pointClouds.size(); i++)
		{
			drawGroups(pointClouds[i], colors[i]);
		}
	#endif 
	#ifndef _DEBUG_INTERACTIVE
		gFrame = gray.clone();
		cv::cvtColor(gFrame, gFrame, CV_GRAY2BGR);
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
	removeFalsePositiveCircles(circles);
	end = std::chrono::high_resolution_clock::now();
	auto durationDetection = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "Time elapsed: " << durationDetection / 1e6 << "ms" << std::endl;
	#ifdef _BENCHMARK
		std::cout << "\tTime to sample: " << (gTimeSample1 + gTimeSample2) / 1e6 << "ms (" << (gTimeSample1 + gTimeSample2) / durationDetection * 100 << "%)" << std::endl
						<< "\t\tFirst point: " << gTimeSample1 / 1e6 << "ms (" << gTimeSample1 / durationDetection * 100 << "%); Second point: " << gTimeSample2 / 1e6 << "ms (" << gTimeSample2 / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to calculate intersection: " << gTimeIntersection / 1e6 << "ms (" << gTimeIntersection / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to add intersection: " << gTimeAddIntersection / 1e6 << "ms (" << gTimeAddIntersection / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to visit cell: " << gTimeVisit / 1e6 << "ms (" << gTimeVisit / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to add circle: " << gTimeAddCircle / 1e6 << "ms (" << gTimeAddCircle / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to remove false positives: " << gTimeRemoveFalsePositive / 1e6 << "ms (" << gTimeRemoveFalsePositive / durationDetection * 100 << "%)" << std::endl
					<< "\tExplained: " << (gTimeSample1 + gTimeSample2 + gTimeIntersection + gTimeAddIntersection + gTimeVisit + gTimeAddCircle + gTimeRemoveFalsePositive) / durationDetection * 100 << "%" << std::endl;
	#endif 
	std::cout << "Num circles: " << circles.size() << std::endl;
	std::cout << "Total time elapsed: " << (durationPreprocessing + durationDetection) / 1e6 << "ms" << std::endl;
	// Draw circles 
	for (const Circle &circle : circles)
	{
		if (!circle.removed)
		{
			drawCircle(circle, cv::Scalar(0, 0, 0), 4);
		}
	}
	cv::Mat img = gFrame.clone();
	for (const Circle &circle : circles)
	{
		if (circle.removed)
		{
			drawCircle(circle, cv::Scalar(0, 0, 255), 2);
		}
		else
		{
			drawCircle(circle, cv::Scalar(255, 255, 255), 2);
		}
	}
	// Display image
	cv::imshow(gEdgeWindowName, gFrame);
	cv::waitKey(0);
	// Display image without false circles 
	gFrame = img;
	for (const Circle &circle : circles)
	{
		if (!circle.removed)
		{
			drawCircle(circle, cv::Scalar(255, 255, 255), 2);
		}
	}
	cv::imshow(gEdgeWindowName, gFrame);
	// Delete data
	delete cell;
	// OpenCV Hough Transform
	begin = std::chrono::high_resolution_clock::now();
	std::vector<cv::Vec3f> cvCircles;
	//cv::HoughCircles(gImgGray, cvCircles, CV_HOUGH_GRADIENT, 1, 20, gCannyLowThreshold * 3, 40, 0, 0);
	end = std::chrono::high_resolution_clock::now();
	auto opencvDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "OpenCV num circles: " << cvCircles.size() << std::endl;
	std::cout << "OpenCV time elapsed: " << opencvDuration / 1e6 << "ms" << std::endl;
	cv::Mat cvImg = gray.clone();
	for (size_t i = 0; i < cvCircles.size(); i++)
	{
	   cv::Point center(cvRound(cvCircles[i][0]), cvRound(cvCircles[i][1]));
	   int radius = cvRound(cvCircles[i][2]);
	   // circle center
	   cv::circle(cvImg, center, 3, cv::Scalar(0,255,0), -1, 8, 0);
	   // circle outline
	   cv::circle(cvImg, center, radius, cv::Scalar(0,0,255), 3, 8, 0);
	 }
	// cv::imshow(gEdgeWindowName + "_OpenCV", cvImg);
}

int main(int argc, char **argv)
{
	std::srand(time(NULL));

	if (argc < 5)
	{
		std::cerr << "Usage: ARHT <INPUT_IMAGE> <CANNY_MIN_THRESHOLD> <CELL_BRANCHING_FACTOR> <NUM_ANGLES> <MIN_ARC_LENGTH>" << std::endl;
		return -1;
	}
	std::string inputFilename(argv[1]);
	gImg = cv::imread(inputFilename);
	cv::cvtColor(gImg, gImgGray, CV_BGR2GRAY);
	gCannyLowThreshold = atoi(argv[2]);
	gCellBranchingFactor = atoi(argv[3]);
	gNumAngles = atoi(argv[4]);
	gMinArcLength = atoi(argv[5]);
	
	if (!gImg.data)
	{
		std::cerr << "ERROR: Could not read image." << std::endl;
		return -1;
	}
	cv::namedWindow(gEdgeWindowName, CV_WINDOW_AUTOSIZE);
	cv::createTrackbar("Min threshold:", gEdgeWindowName, &gCannyLowThreshold, 100, cannyCallback);
	cannyCallback(gCannyLowThreshold, NULL);
	//cv::imshow("Original image", gImg);
	cv::waitKey(0);

	return 0;
}
