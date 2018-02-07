#ifndef BENCHMARK_H
#define BENCHMARK_H

#define _BENCHMARK

#include <chrono>

extern double gTimeProcessImage;
extern double gTimeCreateConnectedComponents;
extern double gTimeCreatePointCloud;
extern double gTimeCreateSampler;
extern double gTimeSample1;
extern double gTimeSample2;
extern double gTimeIntersection;
extern double gTimeAddIntersection;
extern double gTimeAddIntersectionsChildren;
extern double gTimeAddEllipse;
extern double gTimeRemoveEllipsePoints;
extern double gTimeRemoveFalsePositive;

#endif // BENCHMARK_H
