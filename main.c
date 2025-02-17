#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist/mnist.h"
#include <time.h>

#define LAYER_COUNT 4
#define LAYER_NEURON_COUNT (int[]){784, 16, 16, 10}

#define EPOCHS 20
#define MINI_BATCH_SIZE 100
#define LEARNING_RATE 3

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

double sigmoid(double x)
{
    return (1.0 / (1.0 + exp(-x)));
}

double sigmoid_derivative(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

double sigmoid_inverse(double x)
{
    return logl(x / (1.0 - x));
}

void feedforward(double *input, nn_parameters_t *nn_parameters, feedforward_result_t *result)
{
    for (int i = 0; i < LAYER_NEURON_COUNT[0]; i++)
    {
        result->activation_matrices[0][i] = input[i];
        result->z_matrices[0][i] = sigmoid_inverse(input[i]);
    }

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
    }
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

void backprop(double **activation_matrices, double **z_matrices, double **weight_matrices, double *expected_result, backprop_result_t *result)
{
    double **ca_matrices = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));
    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        ca_matrices[i] = (double *)calloc(LAYER_NEURON_COUNT[i + 1], sizeof(double));
    }

    for (int i = 0; i < LAYER_NEURON_COUNT[LAYER_COUNT - 1]; i++)
    {
        ca_matrices[LAYER_COUNT - 2][i] = 2 * (activation_matrices[LAYER_COUNT - 1][i] - expected_result[i]) *
                                          sigmoid_derivative(z_matrices[LAYER_COUNT - 1][i]);
    }
    for (int i = LAYER_COUNT - 2; i >= 1; i--)
    {
        size_t cur_neuron_count = LAYER_NEURON_COUNT[i];
        size_t next_neuron_count = LAYER_NEURON_COUNT[i + 1];

        for (int k = 0; k < cur_neuron_count; k++)
        {
            double sum = 0;
            for (int j = 0; j < next_neuron_count; j++)
            {
                int index = cur_neuron_count * j + k;
                sum += weight_matrices[i][index] * ca_matrices[i][j];
            }
            ca_matrices[i - 1][k] = sum * sigmoid_derivative(z_matrices[i][k]);
        }
    }

    for (int i = LAYER_COUNT - 2; i >= 1; i--)
    {
        size_t next_neuron_count = LAYER_NEURON_COUNT[i + 1];
        size_t cur_neuron_count = LAYER_NEURON_COUNT[i];

        for (int k = 0; k < cur_neuron_count; k++)
        {
            result->bias_gradients[i - 1][k] += ca_matrices[i - 1][k];
            for (int j = 0; j < next_neuron_count; j++)
            {
                int index = cur_neuron_count * j + k;
                result->weight_gradients[i][index] += activation_matrices[i][k] * ca_matrices[i][j];
            }
        }
    }

    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        free(ca_matrices[i]);
    }
    free(ca_matrices);
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

void consume_backprop_result(nn_parameters_t *nn_parameters, backprop_result_t *bp_result)
{
    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];

        for (int j = 0; j < weights_size; j++)
        {
            nn_parameters->weight_matrices[i][j] -= LEARNING_RATE * bp_result->weight_gradients[i][j] / MINI_BATCH_SIZE;
            bp_result->weight_gradients[i][j] = 0;
        }

        for (int j = 0; j < bias_size; j++)
        {
            nn_parameters->bias_matrices[i][j] -= LEARNING_RATE * bp_result->bias_gradients[i][j] / MINI_BATCH_SIZE;
            bp_result->bias_gradients[i][j] = 0;
        }
    }
}

double gaussian_random()
{
    // implementation of box-muller transform

    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return z0;
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
        for (int j = 0; j < weights_size; j++)
            nn_parameters->weight_matrices[i][j] = gaussian_random();
        nn_parameters->bias_matrices[i] = (double *)calloc(bias_size, sizeof(double));
        for (int j = 0; j < bias_size; j++)
            nn_parameters->bias_matrices[i][j] = gaussian_random();
    }

    feedforward_result_t *ff_result = (feedforward_result_t *)calloc(1, sizeof(feedforward_result_t));

    ff_result->activation_matrices = (double **)calloc(LAYER_COUNT, sizeof(double *));
    ff_result->z_matrices = (double **)calloc(LAYER_COUNT, sizeof(double *));

    for (int i = 0; i < LAYER_COUNT; i++)
    {
        ff_result->activation_matrices[i] = (double *)calloc(LAYER_NEURON_COUNT[i], sizeof(double));
        ff_result->z_matrices[i] = (double *)calloc(LAYER_NEURON_COUNT[i], sizeof(double));
    }

    backprop_result_t *bp_result = (backprop_result_t *)calloc(1, sizeof(backprop_result_t));

    bp_result->weight_gradients = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));
    bp_result->bias_gradients = (double **)calloc(LAYER_COUNT - 1, sizeof(double *));

    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];

        bp_result->weight_gradients[i] = (double *)calloc(weights_size, sizeof(double));
        bp_result->bias_gradients[i] = (double *)calloc(bias_size, sizeof(double));
    }

    for (int e = 0; e < EPOCHS; e++)
    {
        for (int i = NUM_TRAIN - 1; i >= 0; i--)
        {
            int idx1 = i;
            int idx2 = rand() % (i + 1);

            for (int j = 0; j < 784; j++)
            {
                double temp_pixel = train_image[idx1][j];
                train_image[idx1][j] = train_image[idx2][j];
                train_image[idx2][j] = temp_pixel;
            }

            int temp_label = train_label[idx1];
            train_label[idx1] = train_label[idx2];
            train_label[idx2] = temp_label;
        }

        int iter = NUM_TRAIN / MINI_BATCH_SIZE;
        for (int l = 0; l < iter; l++)
        {
            for (int i = 0; i < MINI_BATCH_SIZE; i++)
            {
                int idx = MINI_BATCH_SIZE * l + i;
                double expected[LAYER_NEURON_COUNT[LAYER_COUNT - 1]];
                for (int j = 0; j < LAYER_NEURON_COUNT[LAYER_COUNT - 1]; j++)
                {
                    expected[j] = j == train_label[idx] ? 1 : 0;
                }

                feedforward(train_image[idx], nn_parameters, ff_result);

                backprop(
                    ff_result->activation_matrices,
                    ff_result->z_matrices,
                    nn_parameters->weight_matrices,
                    expected,
                    bp_result);
            }
            consume_backprop_result(nn_parameters, bp_result);
        }

        int correct = 0;
        for (int i = 0; i < NUM_TEST; i++)
        {
            feedforward(test_image[i], nn_parameters, ff_result);
            double max = -10.0;
            int max_index = -1;
            for (int i = 0; i < LAYER_NEURON_COUNT[LAYER_COUNT - 1]; i++)
            {
                double d = ff_result->activation_matrices[LAYER_COUNT - 1][i];
                if (d > max)
                {
                    max = d;
                    max_index = i;
                }
            }
            if (test_label[i] == max_index)
            {
                correct += 1;
            }
        }

        printf("completed epoch %d, accuracy : %.2f%%\n", e + 1, 100.0 * (float)correct / (float)NUM_TEST);
    }

    int correct = 0;
    for (int i = 0; i < NUM_TEST; i++)
    {
        feedforward(test_image[i], nn_parameters, ff_result);
        double max = 0;
        int max_index = 0;
        for (int i = 0; i < LAYER_NEURON_COUNT[LAYER_COUNT - 1]; i++)
        {
            double d = ff_result->activation_matrices[LAYER_COUNT - 1][i];
            if (d > max)
            {
                max = d;
                max_index = i;
            }
        }
        if (test_label[i] == max_index)
        {
            correct += 1;
        }
    }
    printf("accuracy: %f\n", 100.0 * (float)correct / (float)NUM_TEST);

    free_feedforward_result(ff_result);
    free_backprop_result(bp_result);

    return 0;
}