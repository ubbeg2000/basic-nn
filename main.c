#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist/mnist.h"

#define LAYER_COUNT 4
#define LAYER_NEURON_COUNT (int[]){784, 16, 16, 10}
#define EULER_NUMBER 2.71828182845904523536

#define ASCII_CHARSET_LEN 71
#define ASCII_CHARSET_MAX_VALUE (ASCII_CHARSET_LEN - 1)
#define ASCII_CHARSET "$@B%%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

typedef struct backprop_result
{
    double **weight_gradients;
    double **bias_gradients;
} backprop_result_t;

typedef struct feedforward_result
{
    double **activation_matrices;
    double **z_matrices;
    double cost;
} feedforward_result_t;

void print_image(int label, double *image)
{
    printf("label: %d\n", label);
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            printf("%c", ASCII_CHARSET[(int)roundl(image[28 * i + j] * ASCII_CHARSET_MAX_VALUE)]);
        }
        printf("\n");
    }
}

double sigmoid(double n)
{
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

double sigmoid_derivative(double n)
{
    return sigmoid(n) * (1 - sigmoid(n));
}

double sigmoid_inverse(double n)
{
    return logl(n / (1 - n));
}

feedforward_result_t *feedforward(double *input, double **weight_matrices, double **bias_matrices)
{
    feedforward_result_t *result = (feedforward_result_t *)calloc(1, sizeof(feedforward_result_t));

    result->activation_matrices = (double **)calloc(LAYER_COUNT, sizeof(double *));
    result->z_matrices = (double **)calloc(LAYER_COUNT, sizeof(double *));

    for (int i = 0; i < LAYER_COUNT; i++)
    {
        result->activation_matrices[i] = (double *)calloc(LAYER_NEURON_COUNT[i], sizeof(double));
        result->z_matrices[i] = (double *)calloc(LAYER_NEURON_COUNT[i], sizeof(double));
    }

    for (int i = 0; i < LAYER_NEURON_COUNT[0]; i++)
    {
        result->activation_matrices[0][i] = input[i];
        result->z_matrices[0][i] = sigmoid_inverse(input[i]);
    }

    print_image(3, input);

    for (int i = 1; i < LAYER_COUNT; i++)
    {
        size_t cur_neuron_count = LAYER_NEURON_COUNT[i - 1];
        size_t next_neuron_count = LAYER_NEURON_COUNT[i];
        for (int j = 0; j < next_neuron_count; j++)
        {
            double mac_result = 0, p, l;
            for (int k = 0; k < cur_neuron_count; k++)
            {
                p = result->activation_matrices[i - 1][k];
                l = weight_matrices[i - 1][cur_neuron_count * j + k];
                mac_result += p * l;
            }
            mac_result += bias_matrices[i - 1][j];
            result->z_matrices[i][j] = mac_result;
            result->activation_matrices[i][j] = sigmoid(mac_result);
        }

        printf("layer: %d\n", i);
        for (int j = 0; j < LAYER_NEURON_COUNT[i]; j++)
        {
            printf("%f\n", result->activation_matrices[i][j]);
        }
    }
    return result;
}

backprop_result_t *backprop(double **activation_matrices, double **z_matrices, double **weight_matrices, double *bias_matrix, double *expected_result)
{
    backprop_result_t *result = (backprop_result_t *)calloc(1, sizeof(backprop_result_t));

    result->weight_gradients = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));
    result->bias_gradients = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));

    double **a_matrices = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));

    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];

        result->weight_gradients[i] = (double *)calloc(weights_size, sizeof(double));
        result->bias_gradients[i] = (double *)calloc(bias_size, sizeof(double));
        a_matrices[i] = (double *)calloc(bias_size, sizeof(double));
    }

    for (int i = 0; i < LAYER_NEURON_COUNT[LAYER_COUNT - 1]; i++)
    {
        a_matrices[LAYER_COUNT - 2][i] = powl(activation_matrices[LAYER_COUNT - 1][i] - expected_result[i], 2);
    }

    for (int i = LAYER_COUNT - 2; i >= 0; i--)
    {
        size_t cur_neuron_count = LAYER_NEURON_COUNT[i - 1];
        size_t next_neuron_count = LAYER_NEURON_COUNT[i];
        for (int j = 0; j < cur_neuron_count; j++)
        {
            for (int k = 0; k < next_neuron_count; k++)
            {
            }
        }
    }

    return NULL;
}

int main()
{
    load_mnist();

    double **weight_matricies = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));
    double **bias_matricies = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));

    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];

        weight_matricies[i] = (double *)calloc(weights_size, sizeof(double));
        bias_matricies[i] = (double *)calloc(weights_size, sizeof(double));
    }

    feedforward_result_t *ff_result = feedforward(train_image[0], weight_matricies, bias_matricies);

    backprop_result_t *bp_result = backprop(
        ff_result->activation_matrices,
        ff_result->z_matrices,
        weight_matricies, bias_matricies, NULL);

    return 0;
}