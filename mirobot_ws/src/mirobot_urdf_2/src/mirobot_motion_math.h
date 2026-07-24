#ifndef MIROBOT_MOTION_MATH_H
#define MIROBOT_MOTION_MATH_H

#include <cmath>
#include <vector>

inline double shortestAngularDistance(double from, double to)
{
	const double two_pi = 2.0 * M_PI;
	double delta = std::fmod(to - from, two_pi);
	if (delta > M_PI)
	{
		delta -= two_pi;
	}
	else if (delta < -M_PI)
	{
		delta += two_pi;
	}
	return delta;
}

inline double accumulatedJointTravel(
	const std::vector<double> &positions, double start)
{
	double travel = 0.0;
	double previous = start;
	for (size_t index = 0; index < positions.size(); ++index)
	{
		travel += std::fabs(positions[index] - previous);
		previous = positions[index];
	}
	return travel;
}

#endif
