#include "circlefunctor.h"

CircleFunctor::CircleFunctor(const Eigen::Matrix2Xf &points)
	: Functor<float>(2, points.cols())
	, mPoints(points)
{

}

int CircleFunctor::operator()(const Eigen::VectorXf &params, Eigen::VectorXf &residuals)
{
	Eigen::Vector2f center(params(0), params(1));
	float radius = std::abs(params(2));
	float radiusSquared = radius * radius;
	residuals = ((mPoints.colwise() - center).colwise().squaredNorm().array() - radiusSquared).pow(2);
	return 0;
}

int CircleFunctor::df(const Eigen::VectorXf &params, Eigen::MatrixXf &jacobian)
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
