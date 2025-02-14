#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist/mnist.h"
#include <time.h>

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

typedef struct nn_parameters
{
    double **weight_matrices;
    double **bias_matrices;
} nn_parameters_t;

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

feedforward_result_t *feedforward(double *input, nn_parameters_t *nn_parameters)
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

    // print_image(3, input);

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
                l = nn_parameters->weight_matrices[i - 1][cur_neuron_count * j + k];
                mac_result += p * l;
            }
            mac_result += nn_parameters->bias_matrices[i - 1][j];
            result->z_matrices[i][j] = mac_result;
            result->activation_matrices[i][j] = sigmoid(mac_result);
        }

        // printf("layer: %d\n", i);
        // for (int j = 0; j < LAYER_NEURON_COUNT[i]; j++)
        // {
        //     printf("%f\n", result->activation_matrices[i][j]);
        // }
    }
    return result;
}

void free_feedforward_result(feedforward_result_t *ff_result)
{
    for (int i = 0; i < LAYER_COUNT; i++)
    {
        free(ff_result->activation_matrices[i]);
        free(ff_result->z_matrices[i]);
    }
    free(ff_result->activation_matrices);
    free(ff_result->z_matrices);
    free(ff_result);
}

backprop_result_t *backprop(double **activation_matrices, double **z_matrices, double **weight_matrices, double *expected_result)
{
    backprop_result_t *result = (backprop_result_t *)calloc(1, sizeof(backprop_result_t));

    result->weight_gradients = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));
    result->bias_gradients = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));

    double **ca_matrices = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));

    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];

        result->weight_gradients[i] = (double *)calloc(weights_size, sizeof(double));
        result->bias_gradients[i] = (double *)calloc(bias_size, sizeof(double));
        ca_matrices[i] = (double *)calloc(bias_size, sizeof(double));
    }

    for (int i = 0; i < LAYER_NEURON_COUNT[LAYER_COUNT - 1]; i++)
    {
        ca_matrices[LAYER_COUNT - 2][i] = 2 * (activation_matrices[LAYER_COUNT - 1][i] - expected_result[i], 2);
    }
    for (int i = LAYER_COUNT - 2; i >= 1; i--)
    {
        size_t cur_neuron_count = LAYER_NEURON_COUNT[i + 1];
        size_t prev_neuron_count = LAYER_NEURON_COUNT[i];
        for (int k = 0; k < prev_neuron_count; k++)
        {
            double sum = 0;
            for (int j = 0; j < cur_neuron_count; j++)
            {
                sum += weight_matrices[i][cur_neuron_count * j + k] * sigmoid_derivative(z_matrices[i][j]) * ca_matrices[i][j];
            }
            ca_matrices[i - 1][k] = sum;
        }
    }

    for (int i = LAYER_COUNT - 2; i >= 1; i--)
    {
        size_t cur_neuron_count = LAYER_NEURON_COUNT[i + 1];
        size_t prev_neuron_count = LAYER_NEURON_COUNT[i];
        for (int j = 0; j < cur_neuron_count; j++)
        {
            for (int k = 0; k < prev_neuron_count; k++)
            {
                int index = cur_neuron_count * j + k;
                result->bias_gradients[i - 1][k] = sigmoid_derivative(z_matrices[i][k]) * ca_matrices[i][k];
                result->weight_gradients[i - 1][index] = activation_matrices[i][k] * result->bias_gradients[i][k];
            }
        }
    }

    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        free(ca_matrices[i]);
    }
    free(ca_matrices);

    return result;
}

void free_backprop_result(backprop_result_t *bp_result)
{
    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        free(bp_result->bias_gradients[i]);
        free(bp_result->weight_gradients[i]);
    }
    free(bp_result->bias_gradients);
    free(bp_result->weight_gradients);
    free(bp_result);
}

void apply_backprop_result(nn_parameters_t *nn_parameters, backprop_result_t *bp_result)
{
    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];

        for (int j = 0; j < weights_size; j++)
        {
            nn_parameters->weight_matrices[i][j] += bp_result->weight_gradients[i][j];
        }

        for (int j = 0; j < bias_size; j++)
        {
            nn_parameters->bias_matrices[i][j] += bp_result->bias_gradients[i][j];
        }
    }
}

double *label_to_expected_result(int label)
{
    double *res = (double *)calloc(LAYER_NEURON_COUNT[LAYER_COUNT - 1], sizeof(double));
    res[label - 1] = 1;
    return res;
}

void print_backprop_result(backprop_result_t *bp_result) {
    FILE *fp = fopen("backprop_result.dat", "w");
    for (int i = 0; i < LAYER_COUNT-1;i++) {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];
        fprintf(fp, "layer %d\n", i+1);
        for (int j = 0; j < weights_size; j++)
        {
            fprintf(fp, "w:%f\n", bp_result->weight_gradients[i][j]);
        }

        for (int j = 0; j < bias_size; j++)
        {
            fprintf(fp, "b:%f\n", bp_result->bias_gradients[i][j]);
        }
    }
    fclose(fp);
}

int main()
{
    srand(time(NULL));
    load_mnist();

    nn_parameters_t *nn_parameters = (nn_parameters_t *)calloc(1, sizeof(nn_parameters_t));

    nn_parameters->weight_matrices = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));
    nn_parameters->bias_matrices = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));

    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];

        nn_parameters->weight_matrices[i] = (double *)calloc(weights_size, sizeof(double));
        for (int j = 0; j < weights_size; j++) nn_parameters->weight_matrices[i][j] = (float)rand()/(float)(RAND_MAX);
        nn_parameters->bias_matrices[i] = (double *)calloc(bias_size, sizeof(double));
        for (int j = 0; j < bias_size; j++) nn_parameters->bias_matrices[i][j] = (float)rand()/(float)(RAND_MAX);
    }

    for (int i = 0; i < 100; i++)
    {
        feedforward_result_t *ff_result = feedforward(train_image[i], nn_parameters);

        backprop_result_t *bp_result = backprop(
            ff_result->activation_matrices,
            ff_result->z_matrices,
            nn_parameters->weight_matrices,
            label_to_expected_result(train_label[i])
        );

        apply_backprop_result(nn_parameters, bp_result);

        free_feedforward_result(ff_result);
        free_backprop_result(bp_result);
    }

    // print_backprop_result(bp_result);

    feedforward_result_t *ff_result = feedforward(train_image[0], nn_parameters);
    for (int i = 0; i < LAYER_NEURON_COUNT[LAYER_COUNT - 1]; i++)
    {
        double d = ff_result->activation_matrices[LAYER_COUNT - 1][i];
        printf("%d: %f\n", i, d);
    }

    return 0;
}