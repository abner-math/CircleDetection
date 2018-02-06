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

#define NUM_ANGLES 60
#define MIN_NUM_ANGLES 15 // 45 degrees 

std::string gEdgeWindowName = "Edge image";
int gCannyLowThreshold;
cv::Mat gImg;
cv::Mat gImgGray;
cv::Mat gFrame;
int gMaxDepth;

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
double gTimeAddIntersectionsChildren;
double gTimeAddEllipse;
double gTimeRemoveFalsePositive;
double gTimeRemoveEllipsePoints;
#endif 

#ifdef _DEBUG_INTERACTIVE
struct Intersection;
struct Ellipse;
struct HoughCell;
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

void drawEllipse(const Ellipse &ellipse, cv::Scalar color = cv::Scalar(255, 255, 255), int thickness = 2)
{
	cv::ellipse(gFrame, ellipse.ellipse, color, thickness);
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
void drawCells(const HoughCell *cell, HoughAccumulator *accumulator)
{
	if (!cell->isVisited())
	{
		if (cell == accumulator->cell())
		{
			if (accumulator->hasEllipseCandidate())
			{
				drawRect(cell->extension(), cv::Scalar(255, 0, 0));
			}
			else
			{
				drawRect(cell->extension(), cv::Scalar(0, 255, 0));
			}
		}
		else
		{
			drawRect(cell->extension(), cv::Scalar(0, 0, 255));
		}
		for (size_t i = 0; i < cell->numAccumulators(); i++)
		{
			if (cell->accumulator(i) != NULL)
			{
				for (size_t j = 0; j < cell->accumulator(i)->intersections().size(); j++)
				{
					const Intersection &intersection = cell->accumulator(i)->intersections()[j];
					cv::Scalar color;
					if (cell == accumulator->cell() && j == cell->accumulator(i)->intersections().size() - 1)
						color = cv::Scalar(0, 255, 0);
					else 
						color = cv::Scalar(0, 0, 255);
					const Point &p1 = intersection.sampler->pointCloud().group(intersection.p1);
					const Point &p2 = intersection.sampler->pointCloud().group(intersection.p2);
					drawLine(p1.position, p1.position + p1.normal * 10, color);
					drawLine(p2.position, p2.position + p2.normal * 10, color);
					drawPoint(intersection.position, color);
				}
			}
		}
	}
	else
	{
		drawRect(cell->extension(), cv::Scalar(0, 0, 255));
		for (size_t i = 0; i < 4; i++)
		{
			if (cell->child(i) != NULL)
			{
				drawCells(cell->child(i), accumulator);
			}
		}
	}
}

void displayInteractiveFrame(const std::vector<PointCloud> &pointClouds, const HoughCell *cell, HoughAccumulator *accumulator, const std::vector<Ellipse> &ellipses)
{
	cv::Mat img = gFrame.clone();
	for (const PointCloud &pointCloud : pointClouds)
	{
		Sampler *sampler = pointCloud.sampler();
		for (size_t i = 0; i < sampler->numPoints(); i++)
		{
			if (sampler->isRemoved(i))
			{
				drawPoint(pointCloud.group(i).position, cv::Scalar(0, 0, 255));
			}
			else
			{
				drawPoint(pointCloud.group(i).position, cv::Scalar(255, 255, 255));
			}
		}
	}
	drawCells(cell, accumulator);
	for (const Ellipse &ellipse : ellipses)
	{
		if (ellipse.falsePositive)
		{
			drawEllipse(ellipse, cv::Scalar(0, 0, 255));
		}
		else
		{
			drawEllipse(ellipse, cv::Scalar(255, 0, 0));
		}
	}
	cv::imshow(gEdgeWindowName, gFrame);
	cv::waitKey(0);
	gFrame = img;
}
#endif 

cv::Rect2i getEllipseBoundingRect(const cv::Mat &edgeImg, const Ellipse &ellipse, int strokeSize)
{
	cv::Rect2i boundingRect = ellipse.ellipse.boundingRect();
	return boundingRect & cv::Rect(0, 0, edgeImg.cols, edgeImg.rows);
}

cv::RotatedRect getCenteredEllipse(const cv::Rect2i &boundingRect, const Ellipse &ellipse)
{
	cv::Point2i newCenter(boundingRect.width / 2, boundingRect.height / 2);
	cv::RotatedRect newEllipse(newCenter, ellipse.ellipse.size, ellipse.ellipse.angle);
	return newEllipse;
}

size_t removeEllipsePoints(const HoughCell *cell, const Ellipse &ellipse, const std::vector<PointCloud> &pointClouds)
{
	cv::Mat edgeImg = PointCloud::edgeImg();
	cv::Rect2i boundingRect = getEllipseBoundingRect(edgeImg, ellipse, (int)cell->size());
	cv::Mat croppedEdgeImg = edgeImg(boundingRect);
	cv::Mat referenceImg = cv::Mat::zeros(boundingRect.size(), CV_8U);
	cv::RotatedRect newEllipse = getCenteredEllipse(boundingRect, ellipse);
	cv::ellipse(referenceImg, newEllipse, cv::Scalar(255, 255, 255), std::max((int)std::min(ellipse.ellipse.size.width, ellipse.ellipse.size.height) / 10, (int)cell->size()));
	cv::Mat toBeRemoved = referenceImg & croppedEdgeImg;
	size_t count = 0;
	int index = 0;
	for (int y = 0; y < toBeRemoved.rows; y++)
	{
		for (int x = 0; x < toBeRemoved.cols; x++)
		{
			if (toBeRemoved.data[index])
			{
				int x_ = x + boundingRect.x;
				int y_ = y + boundingRect.y;
				int index_ = y_ * edgeImg.cols + x_;
				Point *point = PointCloud::points()[index_];
				if (point->isGroup && !point->pointCloud->sampler()->isRemoved(point->index))
				{
					point->pointCloud->sampler()->removePoint(point->index);
					++count;
				}
			}
			++index;
		} 
	}
	return count;
}

bool isFalsePositive(Ellipse &ellipse)
{
	if (ellipse.falsePositive) return true;
	cv::Mat edgeImg = PointCloud::edgeImg();
	cv::Rect2i boundingRect = getEllipseBoundingRect(edgeImg, ellipse, 4);
	float a = ellipse.ellipse.size.width / 2;    
	float b = ellipse.ellipse.size.height / 2;
	float eccentricity = std::sqrt(1 - (a * a) / (b * b));
	if (eccentricity > 0.75f) return true;
	if (std::max(boundingRect.width, boundingRect.height) / (float)std::min(boundingRect.width, boundingRect.height) > 2) return false;
	cv::Mat croppedEdgeImg = edgeImg(boundingRect);
	cv::Mat referenceImg = cv::Mat::zeros(boundingRect.size(), CV_8U);
	cv::RotatedRect newEllipse = getCenteredEllipse(boundingRect, ellipse);
	cv::ellipse(referenceImg, newEllipse, cv::Scalar(255, 255, 255), 2);
	std::vector<std::vector<cv::Point> > contours; 
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(referenceImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	if (contours.size() < 2) return true;
	double perimeter = (cv::arcLength(contours[0], true) + cv::arcLength(contours[1], true)) / 2;
	cv::Mat resultImg = referenceImg & croppedEdgeImg;
	double value = cv::sum(resultImg)[0] / 255.0 / perimeter;
	return value < 0.5f;
}

void findEllipses(HoughCell *cell, const std::vector<PointCloud> &pointClouds, std::vector<Ellipse> &ellipses);

void subdivide(HoughCell *parentCell, HoughAccumulator *accumulator, const std::vector<PointCloud> &pointClouds, std::vector<Ellipse> &ellipses)
{
	HoughCell *cell = accumulator->cell();
	cell->setVisited();
	if (cell->depth() < gMaxDepth)
	{
		for (HoughAccumulator *childAccumulator : cell->addIntersectionsToChildren())
		{
			#ifdef _DEBUG_INTERACTIVE
				displayInteractiveFrame(pointClouds, parentCell, accumulator, ellipses);
			#endif
			subdivide(parentCell, childAccumulator, pointClouds, ellipses);
		}
		//findEllipses(parentCell, pointClouds, ellipses);
	}        
	else
	{
		#ifdef _BENCHMARK
			auto begin = std::chrono::high_resolution_clock::now();
		#endif
		Ellipse ellipse = accumulator->getEllipseCandidate();
		#ifdef _BENCHMARK
			auto end = std::chrono::high_resolution_clock::now();
			gTimeAddEllipse += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
			begin = std::chrono::high_resolution_clock::now();
		#endif 
		ellipse.falsePositive = isFalsePositive(ellipse);
		#ifdef _BENCHMARK
			end = std::chrono::high_resolution_clock::now();
			gTimeRemoveFalsePositive += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
			begin = std::chrono::high_resolution_clock::now();
		#endif
		if (!ellipse.falsePositive)
		{
			//adjustEllipse(cell, ellipse, pointClouds);
			removeEllipsePoints(cell, ellipse, pointClouds);
			#ifdef _BENCHMARK
				end = std::chrono::high_resolution_clock::now();
				gTimeRemoveEllipsePoints += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
			#endif
			ellipses.push_back(ellipse);
			#ifdef _DEBUG_INTERACTIVE
				displayInteractiveFrame(pointClouds, parentCell, accumulator, ellipses);
			#endif
		}
		else
		{
			#ifdef _DEBUG_INTERACTIVE
				ellipses.push_back(ellipse);
				displayInteractiveFrame(pointClouds, parentCell, accumulator, ellipses);
				ellipses.erase(ellipses.begin() + ellipses.size() - 1);
			#endif
			//for (const Intersection &intersection : accumulator->intersections())
			//{
				//intersection.sampler->addPoint(intersection.p1);
			//	intersection.sampler->addPoint(intersection.p2);
			//}
			/*for (HoughAccumulator *childAccumulator : cell->addIntersectionsToChildren())
			{
				#ifdef _DEBUG_INTERACTIVE
					displayInteractiveFrame(pointClouds, parentCell, accumulator, ellipses);
				#endif
				subdivide(parentCell, childAccumulator, pointClouds, ellipses);
			}
			findEllipses(parentCell, pointClouds, ellipses);*/
		}	
	}
}
 
void findEllipses(HoughCell *cell, const std::vector<PointCloud> &pointClouds, std::vector<Ellipse> &ellipses)
{
	for (const PointCloud &pointCloud : pointClouds)
	{
		Sampler *sampler = pointCloud.sampler();
		while (sampler->canSample())
		{
			HoughAccumulator *accumulator = cell->addIntersection(sampler);
			if (accumulator != NULL)
			{
				#ifdef _DEBUG_INTERACTIVE
					displayInteractiveFrame(pointClouds, cell, accumulator, ellipses);
				#endif
				if (accumulator->hasEllipseCandidate())
				{
					subdivide(cell, accumulator, pointClouds, ellipses);
				}
			}
		}
	}
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
		gTimeAddIntersectionsChildren = 0;
		gTimeAddEllipse = 0;
		gTimeRemoveEllipsePoints = 0;
		gTimeRemoveFalsePositive = 0;
	#endif 
	std::cout << "Preprocessing..." << std::endl;
	auto begin = std::chrono::high_resolution_clock::now();
	cv::Mat gray = gImgGray.clone();
	std::cout << "Image size: " << gray.size() << std::endl;
	// Calculate normals and curvatures 
	std::vector<PointCloud> pointClouds;
	cv::Rect2f extension = PointCloud::createPointCloudsFromImage(gray, gCannyLowThreshold, NUM_ANGLES, pointClouds);
	std::cout << "Extension: " << extension << std::endl;
	for (PointCloud &pointCloud : pointClouds)
	{
		pointCloud.createSampler(MIN_NUM_ANGLES);
	}
	float size = extension.width / 2;
	cv::Point2f center(extension.x + size, extension.y + size);
	HoughCell *cell = new HoughCell(extension, center, size, NUM_ANGLES, MIN_NUM_ANGLES);
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
	std::vector<Ellipse> ellipses;
	findEllipses(cell, pointClouds, ellipses);
	end = std::chrono::high_resolution_clock::now();
	auto durationDetection = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "Time elapsed: " << durationDetection / 1e6 << "ms" << std::endl;
	#ifdef _BENCHMARK
		std::cout << "\tTime to sample: " << (gTimeSample1 + gTimeSample2) / 1e6 << "ms (" << (gTimeSample1 + gTimeSample2) / durationDetection * 100 << "%)" << std::endl
						<< "\t\tFirst point: " << gTimeSample1 / 1e6 << "ms (" << gTimeSample1 / durationDetection * 100 << "%); Second point: " << gTimeSample2 / 1e6 << "ms (" << gTimeSample2 / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to calculate intersection: " << gTimeIntersection / 1e6 << "ms (" << gTimeIntersection / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to add intersection: " << gTimeAddIntersection / 1e6 << "ms (" << gTimeAddIntersection / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to add intersections to children: " << gTimeAddIntersectionsChildren / 1e6 << "ms (" << gTimeAddIntersectionsChildren / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to add ellipse: " << gTimeAddEllipse / 1e6 << "ms (" << gTimeAddEllipse / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to remove ellipse points: " << gTimeRemoveEllipsePoints / 1e6 << "ms (" << gTimeRemoveEllipsePoints / durationDetection * 100 << "%)" << std::endl
					<< "\tTime to remove false positives: " << gTimeRemoveFalsePositive / 1e6 << "ms (" << gTimeRemoveFalsePositive / durationDetection * 100 << "%)" << std::endl
					<< "\tExplained: " << (gTimeSample1 + gTimeSample2 + gTimeIntersection + gTimeAddIntersection + gTimeAddIntersectionsChildren + gTimeAddEllipse + gTimeRemoveEllipsePoints + gTimeRemoveFalsePositive) / durationDetection * 100 << "%" << std::endl;
	#endif 
	std::cout << "Num ellipses: " << ellipses.size() << std::endl;
	std::cout << "Total time elapsed: " << (durationPreprocessing + durationDetection) / 1e6 << "ms" << std::endl;
	// Draw ellipses  
	for (const Ellipse &ellipse : ellipses)
	{
		//drawEllipse(ellipse, cv::Scalar(0, 0, 0), 4);
		if (!ellipse.falsePositive)
		{
			drawEllipse(ellipse, cv::Scalar(0, 255, 0), 2);
		}
		else
		{
			//drawEllipse(ellipse, cv::Scalar(0, 0, 255), 2);
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
	 //cv::imshow(gEdgeWindowName + "_OpenCV", cvImg);
}

int main(int argc, char **argv)
{
	std::srand(time(NULL));

	if (argc < 4)
	{
		std::cerr << "Usage: ARHT <INPUT_IMAGE> <CANNY_MIN_THRESHOLD> <MAX_DEPTH>" << std::endl;
		return -1;
	}
	std::string inputFilename(argv[1]);
	gImg = cv::imread(inputFilename);
	cv::cvtColor(gImg, gImgGray, CV_BGR2GRAY);
	gCannyLowThreshold = atoi(argv[2]);
	gMaxDepth = atoi(argv[3]);
	
	if (!gImg.data)
	{
		std::cerr << "ERROR: Could not read image." << std::endl;
		return -1;
	}
	cv::namedWindow(gEdgeWindowName, CV_WINDOW_AUTOSIZE);
	//cv::createTrackbar("Min threshold:", gEdgeWindowName, &gCannyLowThreshold, 100, cannyCallback);
	//cv::createTrackbar("Max depth:", gEdgeWindowName, &gMaxDepth, 15, cannyCallback);
	cannyCallback(gCannyLowThreshold, NULL);
	//cv::imshow("Original image", gImg);
	cv::waitKey(0);

	return 0;
}
