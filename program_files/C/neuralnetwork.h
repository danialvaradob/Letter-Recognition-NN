#include <math.h>

// NEURAL NETWORK HYPERPARAMETERS
static const int num_input_nodes = 784;
static const int num_hidden_nodes = 64;
static const int num_output_nodes = 6;
static const int amount_of_batches = 186;
const double learning_rate = 0.50f;

double sigmoid(double x);
double dSigmoid(double x);
double init_weight();
void shuffle(int *array, size_t n);
double* loadWeights(int rows, int cols, char path[], double* weights);
int saveWeights(int rows, int cols, double* weights, char path[]);
double* loadBias(int size, char path[]);
int saveBias(int size, double* bias, char path[]); 