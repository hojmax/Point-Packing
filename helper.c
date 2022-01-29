#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Point {
    float x;
    float y;
} Point;

// Returns the gradients of the points.
Point* get_gradient(Point* points, unsigned int number_of_points, int w, int h,
                    int border_weight) {
    Point* gradient = malloc(number_of_points * sizeof(Point));
#pragma omp parallel for
    for (unsigned int i = 0; i < number_of_points; i++) {
        gradient[i].x = (1 / ((w - points[i].x) * (w - points[i].x)) -
                         1 / (points[i].x * points[i].x)) *
                        border_weight;
        gradient[i].y = (1 / ((h - points[i].y) * (h - points[i].y)) -
                         1 / (points[i].y * points[i].y)) *
                        border_weight;
    }
#pragma omp parallel for
    for (unsigned int i = 0; i < number_of_points; i++) {
        for (unsigned int j = 0; j < number_of_points; j++) {
            if (i == j) continue;
            const float delta_x = points[j].x - points[i].x;
            const float delta_y = points[j].y - points[i].y;
            const float divisor =
                pow(delta_x * delta_x + delta_y * delta_y, 3. / 2.);
            const float g_x = delta_x / divisor;
            const float g_y = delta_y / divisor;
            gradient[i].x += g_x;
            gradient[i].y += g_y;
        }
    }
    return gradient;
}

// Returns clamped value of d between min and max.
float clamp(float d, float min, float max) {
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

// Returns the magnitude of a vector (Reusing the point struct).
float get_magnitude(Point p) { return sqrt(p.x * p.x + p.y * p.y); }

// Limits the magnitude of the gradient and applies the learning rate.
void transform_gradient(Point* gradient, unsigned int number_of_points,
                        float learning_rate, float max_magnitude) {
#pragma omp parallel for
    for (unsigned int i = 0; i < number_of_points; i++) {
        const float magnitude = get_magnitude(gradient[i]);
        if (magnitude <= max_magnitude) {
            gradient[i].x *= learning_rate;
            gradient[i].y *= learning_rate;
        } else {
            gradient[i].x *= learning_rate * max_magnitude / magnitude;
            gradient[i].y *= learning_rate * max_magnitude / magnitude;
        }
    }
}

// Updates the point positions in place.
void update_points(Point* points, unsigned int number_of_points, int w, int h,
                   int border_weight, float min_border_distance,
                   float learning_rate, float max_magnitude) {
    Point* gradient =
        get_gradient(points, number_of_points, w, h, border_weight);
    transform_gradient(gradient, number_of_points, learning_rate,
                       max_magnitude);
#pragma omp parallel for
    for (unsigned int i = 0; i < number_of_points; i++) {
        points[i].x = clamp(points[i].x - gradient[i].x, min_border_distance,
                            w - min_border_distance);
        points[i].y = clamp(points[i].y - gradient[i].y, min_border_distance,
                            h - min_border_distance);
    }
    free(gradient);
}

// Returns random number between 0 and max_value.
float random_float(float max_value) {
    return rand() / (float)(RAND_MAX)*max_value;
}