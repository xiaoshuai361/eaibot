#ifndef MIROBOT_MOTION_MATH_H
#define MIROBOT_MOTION_MATH_H

#include <cmath>
#include <limits>

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

// Select the joint6 equivalent angle nearest to its preceding trajectory point.
inline double nearestEquivalentAngleWithinLimits(
	double target, double reference, double lower, double upper)
{
	const double two_pi = 2.0 * M_PI;
	double best = target;
	double best_distance = std::numeric_limits<double>::infinity();
	for (int turns = -2; turns <= 2; ++turns)
	{
		const double candidate = target + turns * two_pi;
		if (candidate < lower || candidate > upper)
		{
			continue;
		}
		const double distance = std::fabs(candidate - reference);
		if (distance < best_distance)
		{
			best = candidate;
			best_distance = distance;
		}
	}
	return best;
}

#endif
