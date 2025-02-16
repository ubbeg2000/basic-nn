#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist/mnist.h"
#include <time.h>

#define LAYER_COUNT 4
#define LAYER_NEURON_COUNT (int[]){784, 16, 16, 10}
#define EULER_NUMBER 2.71828182845904523536

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

double sigmoid(double x)
{
    return (1.0 / (1.0 + powl(EULER_NUMBER, -x)));
}

double sigmoid_derivative(double x)
{
    // if (n > 1) {
    //     printf("PLER %.10e %.10e %.10e\n", n, sigmoid(n), sigmoid(n) * (1 - sigmoid(n)));
    // }
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
        // printf("PLER %lf", sigmoid_inverse(input[i]));
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
        ca_matrices[LAYER_COUNT - 2][i] = (activation_matrices[LAYER_COUNT - 1][i] - expected_result[i]) *
                                          sigmoid_derivative(z_matrices[LAYER_COUNT - 1][i]);
        // if (ca_matrices[LAYER_COUNT - 2][i] != 0.0) {
        // printf("PLER %.10e %.10e\n", sigmoid_derivative(z_matrices[LAYER_COUNT - 1][i]), ca_matrices[LAYER_COUNT - 2][i]);
        // }
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
                sum += weight_matrices[i][cur_neuron_count * k + j] * ca_matrices[i][j];
                // if (sum == 0) {
                //     printf("PEDAH %.10e %.10e\n", weight_matrices[i][cur_neuron_count * k + j], ca_matrices[i][j]);
                // }
            }
            ca_matrices[i - 1][k] = sum * sigmoid_derivative(z_matrices[i][k]);
            // if (i == 2) {
            //     printf("%d %d %.10e %.10e %.10e %.10e\n", i, k, sum, z_matrices[i][k], ca_matrices[i-1][k], sigmoid_derivative(z_matrices[i][k]));
            // }
        }
        // exit(0);
    }

    for (int i = LAYER_COUNT - 2; i >= 1; i--)
    {
        size_t next_neuron_count = LAYER_NEURON_COUNT[i + 1];
        size_t cur_neuron_count = LAYER_NEURON_COUNT[i];

        for (int k = 0; k < cur_neuron_count; k++)
        {
            result->bias_gradients[i - 1][k] = ca_matrices[i - 1][k];
            for (int j = 0; j < next_neuron_count; j++)
            {
                int index = cur_neuron_count * j + k;
                result->weight_gradients[i][index] += activation_matrices[i][k] * ca_matrices[i][j];
                // if (i == 2) printf("%d %d %d\n", j, k, index);
                // if (result->weight_gradients[i][index] == 0)
                // {
                //     printf("%d,%d,%d : %.10e, %.10e\n", i, j, k, activation_matrices[i][k], ca_matrices[i][j]);
                // }
            }
        }
    }

    // FILE *fp = fopen("mantap.txt", "w");
    // for (int i = 0; i < LAYER_COUNT - 1; i++)
    // {
    //     for (int j = 0; j < LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1]; j++)
    //     {
    //         fprintf(fp, "%.10e ", result->weight_gradients[i][j]);
    //     }
    //     fprintf(fp, "\n");
    // }
    // fclose(fp);

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

double *label_to_expected_result(int label)
{
    double *res = (double *)calloc(LAYER_NEURON_COUNT[LAYER_COUNT - 1], sizeof(double));
    res[label - 1] = 1;
    return res;
}

void print_backprop_result(backprop_result_t *bp_result)
{
    FILE *fp = fopen("backprop_result.dat", "w");
    for (int i = 0; i < LAYER_COUNT - 1; i++)
    {
        size_t bias_size = LAYER_NEURON_COUNT[i + 1];
        size_t weights_size = LAYER_NEURON_COUNT[i] * LAYER_NEURON_COUNT[i + 1];
        fprintf(fp, "layer %d\n", i + 1);
        for (int j = 0; j < weights_size; j++)
        {
            fprintf(fp, "w:%.10e\n", bp_result->weight_gradients[i][j]);
        }

        for (int j = 0; j < bias_size; j++)
        {
            fprintf(fp, "b:%.10e\n", bp_result->bias_gradients[i][j]);
        }
    }
    fclose(fp);
}

void shuffle_training_data(double **train_image, int *train_label)
{
    for (int i = NUM_TRAIN - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        double *temp_image = train_image[i];
        train_image[i] = train_image[j];
        train_image[j] = temp_image;

        int temp_label = train_label[i];
        train_label[i] = train_label[j];
        train_label[j] = temp_label;
    }
}

double gaussian_random() {
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
            nn_parameters->weight_matrices[i][j] = gaussianRandom();
        nn_parameters->bias_matrices[i] = (double *)calloc(bias_size, sizeof(double));
        for (int j = 0; j < bias_size; j++)
            nn_parameters->bias_matrices[i][j] = gaussianRandom();
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

    for (int e = 0; e < 10; e++)
    {
        shuffle_training_data((double **)train_image, train_label);

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

                // if (i == 0)
                // {
                //     exit(0);
                // }
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
                // printf("%d: %f (%d)\n", i, d, train_label[0]);
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

        print_backprop_result(bp_result);
        printf("completed epoch %d, accuracy : %f\n", e + 1, 100.0 * (float)correct / (float)NUM_TEST);
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
            // printf("%d: %f (%d)\n", i, d, train_label[0]);
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