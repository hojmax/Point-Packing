#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Point {
    float x;
    float y;
} Point;

// Helper function for adding to the gradient given two points
void add_point_to_gradient(Point* gradient, Point p1, Point p2, unsigned int i, float weight,
    unsigned int j, int add_both_directions)
{
    const float delta_x = p1.x - p2.x;
    const float delta_y = p1.y - p2.y;
    const float divisor = pow(delta_x * delta_x + delta_y * delta_y, 3. / 2.);
    const float g_x = delta_x / divisor * weight;
    const float g_y = delta_y / divisor * weight;
    gradient[i].x += g_x;
    gradient[i].y += g_y;
    if (!add_both_directions)
        return;
    gradient[j].x -= g_x;
    gradient[j].y -= g_y;
}

// Helper function for adding to the gradient given a point and its bounds.
void add_bounds_to_gradient(Point* gradient, Point* points, unsigned int w, unsigned int h,
    unsigned int i, float border_weight)
{
    gradient[i].x += (1 / ((w - points[i].x) * (w - points[i].x)) - 1 / (points[i].x * points[i].x))
        * border_weight;
    gradient[i].y += (1 / ((h - points[i].y) * (h - points[i].y)) - 1 / (points[i].y * points[i].y))
        * border_weight;
}

// Returns the gradient of the points.
Point* get_gradient(Point* points, unsigned int number_of_points, unsigned int w, unsigned int h,
    float border_weight, Point repel_point, int do_repel, float repel_weight)
{
    Point* gradient = calloc(number_of_points, sizeof(Point));
#pragma omp parallel for
    for (unsigned int i = 0; i < number_of_points; i++) {
        add_bounds_to_gradient(gradient, points, w, h, i, border_weight);
        if (do_repel)
            add_point_to_gradient(gradient, repel_point, points[i], i, repel_weight, 0, 0);
    }
#pragma omp parallel
    {
        Point* private_gradient = calloc(number_of_points, sizeof(Point));
#pragma omp for
        for (unsigned int i = 0; i < number_of_points; i++) {
            for (unsigned int j = i + 1; j < number_of_points; j++) {
                add_point_to_gradient(private_gradient, points[j], points[i], i, 1, j, 1);
            }
        }
#pragma omp critical
        {
            for (unsigned int i = 0; i < number_of_points; i++) {
                gradient[i].x += private_gradient[i].x;
                gradient[i].y += private_gradient[i].y;
            }
        }
    }
    return gradient;
}

// Returns clamped value of d between min and max.
float clamp(float d, float min, float max)
{
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

// Returns the magnitude of a vector (Reusing the point struct).
float get_magnitude(Point p) { return sqrt(p.x * p.x + p.y * p.y); }

// Limits the magnitude of the gradient and applies the learning rate.
void transform_gradient(
    Point* gradient, unsigned int number_of_points, float learning_rate, float max_magnitude)
{
#pragma omp parallel for
    for (unsigned int i = 0; i < number_of_points; i++) {
        gradient[i].x *= learning_rate;
        gradient[i].y *= learning_rate;
        const float magnitude = get_magnitude(gradient[i]);
        if (magnitude >= max_magnitude) {
            gradient[i].x *= max_magnitude / magnitude;
            gradient[i].y *= max_magnitude / magnitude;
        }
    }
}

// Updates the point positions in place.
void update_points(Point* points, unsigned int number_of_points, unsigned int w, unsigned int h,
    float border_weight, float min_border_distance, float learning_rate, float max_magnitude,
    Point repel_point, int do_repel, float repel_weight)
{
    Point* gradient = get_gradient(
        points, number_of_points, w, h, border_weight, repel_point, do_repel, repel_weight);
    transform_gradient(gradient, number_of_points, learning_rate, max_magnitude);
#pragma omp parallel for
    for (unsigned int i = 0; i < number_of_points; i++) {
        points[i].x
            = clamp(points[i].x - gradient[i].x, min_border_distance, w - min_border_distance);
        points[i].y
            = clamp(points[i].y - gradient[i].y, min_border_distance, h - min_border_distance);
    }
    free(gradient);
}
