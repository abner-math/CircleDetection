#ifndef CIRCLE_FUNCTOR_H
#define CIRCLE_FUNCTOR_H

#include "functor.hpp"

class CircleFunctor : public Functor<float>
{
public:
    CircleFunctor(const Eigen::Matrix2Xf &points);

    int operator()(const Eigen::VectorXf &params, Eigen::VectorXf &residuals);

    int df(const Eigen::VectorXf &params, Eigen::MatrixXf &jacobian);

private:
    Eigen::Matrix2Xf mPoints;

};

#endif // CIRCLE_FUNCTOR_H
