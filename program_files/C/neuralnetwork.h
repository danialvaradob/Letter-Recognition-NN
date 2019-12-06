#include <math.h>

// NEURAL NETWORK HYPERPARAMETERS
static const int num_input_nodes = 784;
static const int num_hidden_nodes = 64;
static const int num_output_nodes = 5;
const double lr = 0.1f;

double sigmoid(double x);
double dSigmoid(double x);
double init_weight();
void shuffle(int *array, size_t n);