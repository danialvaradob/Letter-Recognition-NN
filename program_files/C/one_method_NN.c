//
//  
//  NeuralNetwork
//
//  Created by Daniel Alvarado Bonilla
//  Based in:  https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neuralnetwork.h"
//#include "header/mnist_file.h"

#define PIXEL_SCALE(x) (((double) (x)) / 255.0f)


/*
Activation Function
Derivative of AF
*/
double a = 1;
double sigmoid(double x) {
    return 1 / (1 + exp(-x * a)); 
}

double dSigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double init_weight() {
    return ((double)rand())/((double)RAND_MAX); 
}



/*
Method used to shuffle the order of an array
*/
void shuffle(int *array, size_t n) {
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}


/**
 * Method used to load weights from a .dat file into a *double 

 */

double* load_weights(int rows, int cols, char path[], double* weights) {
    
    FILE *file;
    file = fopen(path, "r");
    int i, j;

    fread(&weights[0],sizeof(double), rows * cols , file);

    fclose(file);

    return weights;
}




/* Method used to save weights into a simple dat file
 * Not optimized
 * 
 * "ImageCrop/weightsHidden.dat"
 */

int save_weights(int rows, int cols, double* weights, char path[]) {
    FILE *file;
    file = fopen(path, "w");

    if (file != NULL) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                //fprintf(file, "%lf ", weights[i * cols + j]); 
                fwrite(&weights[i * cols + j] ,sizeof(double), 1 ,file);
            }
        }
    }    
    

    fclose(file);

    return 1;
}

/**
 * Method used to load BIAS from a .dat file into a *double 

 */

double* load_bias(int size, char path[]) {
    
    FILE *file;
    file = fopen(path, "r");
    int i, j;

    double* bias = malloc(sizeof(double) * size );

    fread(&bias[0],sizeof(double),size, file);

    fclose(file);

    return bias;
}




/* Method used to save BIAS into a simple dat file
 * Not optimized
 * 
 * "ImageCrop/bias_hidden.dat"
 * "ImageCrop/bias_output.dat"
 * 
 */

int save_bias(int size, double* bias, char path[]) {
    FILE *file;
    file = fopen(path, "w");

    if (file != NULL) {
        for (int i = 0; i < size; i++) {
            fwrite(&bias[i] ,sizeof(double), 1 ,file);
        }
    }    
    

    fclose(file);

    return 1;
}

/**
 * Method used to test the load/save weights methods. Called upon files.
 * 
 */
void test_w() {
    int r = 4;
    int c = 3;
    double* w = malloc(sizeof(double) * r * c);
    int a = 0;
    char path[] = "ImageCrop/weightsHidden.dat";

   w = load_weights(r, c, path, w);

   

}

/**
 * Given an output of num_output_nodes
 * Returns the Letter with the highest output 
 */ 
char* get_letter(double* output) {
    int letter_pos;
    int highest = 0;

    char* letter;

    for (int i =0; i < num_output_nodes; i ++) {
        if (output[i] > highest) {
            letter_pos = i;
            highest = output[i];
        }
    }

    
    switch( letter_pos )
    {
        case 0:
                letter = "A";
                break;
        case 1:
                letter = "B";
                break;
        case 2:
                letter = "C";
                break;
        case 3:
                letter = "D";
                break;
        case 4:
                letter = "E";
                break;
        case 5:
                letter = "F";
                break;
        case 6:
                letter = " ";
                break;
    }

    return letter;

}

/**
 * Given an output of num_output_nodes
 * Prints letter given by the Neural Network
 */ 
void print_network_output(double* output) {
    int letter_pos;
    double highest = 0;
    for (int i =0; i < num_output_nodes; i ++) {
        //printf("Output is: %f, Highest is: %f\n", output[i], highest);
        if (output[i] > highest) {
            letter_pos = i;
            highest = output[i];

        }
    } 
    switch( letter_pos )
    {
        case 0:
                printf("A");
                break;
        case 1:
                printf("B");
                break;
        case 2:
                printf("C");
                break;
        case 3:
                printf("D");
                break;
        case 4:
                printf("E");
                break;
        case 5:
                printf("F");
                break;
        case 6:
                printf("X");
                break;
    }
}

/**
 * Method used to compare the network output and the label
 * If they're equal, means the network made a correct guess.
 */ 
int correct_guess(double* network_output, double* label, int img ) {
    int correct = 1;
    for (int i = 0; i < num_output_nodes; i++) {
        if (network_output[i] != label[img * num_output_nodes + i ]) {
            correct = 0;
        }
    }
    return correct;
}




/**
 * 
 *  MAIN METHOD.
 *  Used to train the Neural Network.
 *  The rand_flag is used to determine if the weights should be loaded or not
 *  (Not loaded only on the first batch in the first epoc)
 * 
 */ 
int train(double training_inputs[], double training_outputs[], int numTrainingSets, double* hiddenLayerBias, double* outputLayerBias,
                double* hiddenWeights, double* outputWeights) 
    
    {

    double hiddenLayer[num_hidden_nodes];
    double outputLayer[num_output_nodes];

    // array that stores the order of the batch
    int trainingSetOrder[numTrainingSets];

    for (int k =0; k < numTrainingSets; k++) {
        trainingSetOrder[k] = k;
    }
    
    int i;

    int correct = 0;
    int incorrect = 0;

    //printf("Hidden:\n");
    hiddenWeights = load_weights(num_input_nodes, num_hidden_nodes,"W_Data/weightsHidden.dat" , hiddenWeights);
    //printf("Output:\n");
    outputWeights = load_weights(num_hidden_nodes, num_output_nodes, "W_Data/weightsHidden.dat", outputWeights);
    hiddenLayerBias = load_bias(num_hidden_nodes, "W_Data/bias_hidden.dat");
    outputLayerBias = load_bias(num_output_nodes,"W_Data/bias_output.dat");

        // shuffles order
        shuffle(trainingSetOrder,numTrainingSets);

        // iterates through all images of the batch
        for (int x=0; x<numTrainingSets; x++) {
            
            // image selected
            i = trainingSetOrder[x];

            
            for (int j=0; j< num_hidden_nodes; j++) {
                double activation = hiddenLayerBias[j];
                 for (int k=0; k < num_input_nodes; k++) {
                    activation += (PIXEL_SCALE(training_inputs[i*num_input_nodes + k])  * hiddenWeights[k * num_hidden_nodes + j]);
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            //printf("\nNN OUTPUT: ");
            for (int j=0; j<num_output_nodes; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<num_hidden_nodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k * num_output_nodes + j];
                }
                outputLayer[j] = sigmoid(activation);
                //printf("%lf ", outputLayer[j]);
            }



            if ((get_letter(outputLayer)) == (get_letter(training_outputs + (i * num_output_nodes + 0)) )) {
                correct++;
            } else {
                incorrect++;
            }
           
           // Backprop
            double deltaOutput[num_output_nodes];
            for (int j=0; j<num_output_nodes; j++) {

                double errorOutput = (training_outputs[i * num_output_nodes + j] - outputLayer[j]);
                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
            }
            
            double deltaHidden[num_hidden_nodes];
            for (int j=0; j<num_hidden_nodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<num_output_nodes; k++) {
                    errorHidden += deltaOutput[k] * outputWeights[j * num_output_nodes +k];
                }
                deltaHidden[j] = errorHidden * dSigmoid(hiddenLayer[j]);
            }
            
            for (int j=0; j<num_output_nodes; j++) {
                outputLayerBias[j] += deltaOutput[j] * learning_rate;
                for (int k=0; k<num_hidden_nodes; k++) {
                    outputWeights[k * num_output_nodes + j] += hiddenLayer[k] * deltaOutput[j] * learning_rate;
                }
            }
            
            for (int j=0; j<num_hidden_nodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j]*learning_rate;
                for(int k=0; k<num_input_nodes; k++) {
                    //*(training_inputs +i*num_input_nodes + k)
                    hiddenWeights[k *num_hidden_nodes + j] += training_inputs[i*num_input_nodes + k] * deltaHidden[j]*learning_rate;
                }
            }
        }

    
    save_weights(num_input_nodes, num_hidden_nodes, hiddenWeights,"W_Data/weightsHidden.dat");
    save_weights(num_hidden_nodes, num_output_nodes, outputWeights, "W_Data/weightsOutput.dat");
    save_bias(num_hidden_nodes, hiddenLayerBias,"W_Data/bias_hidden.dat" );
    save_bias(num_output_nodes, outputLayerBias,"W_Data/bias_output.dat" );

    
    return correct;

}



/**
 * 
 *  Method used to set the training variable
 *  The path will be batch_name = "batch%d.dat" % (batch_n)
 * 
 * 
 */
double*  set_training_data(double* training_inputs, int numTrainingSets, int inputSize, char path[]) {
    

    FILE *file = NULL;
    file = fopen(path, "r");

    
    int counter = 0;
    int max = numTrainingSets * inputSize;
    while ((!feof(file)) && (counter < max)) {
        fscanf(file, "%lf", &training_inputs[counter]);
        counter++;
        
    }
    
    fclose(file);
    return training_inputs;

}


/**
 * Method used to load training output into a double* variable
 *
 */  
double* set_ouput_data(double* training_outputs ,int numTrainingSets, 
                    int outputSize, char path[]) {
    
    FILE *file = NULL;
    file = fopen(path, "r");

    
    int counter = 0;
    int max = numTrainingSets * num_output_nodes;
    while ((!feof(file)) && (counter < max)) {
        fscanf(file, "%lf", &training_outputs[counter]);
        counter++;
        
    }


    fclose(file);


    return training_outputs;
}



/**
 * Method that loads the input data (images in each batch) and their respective output (labels)
 * 
 */ 
int execute_training(int numTrainingSets, char path_td[], char path_od[], double* hiddenWeights, 
             double* outputWeights, double* hiddenLayerBias, double* outputLayerBias) {


    double* training_inputs = malloc(sizeof(double) * numTrainingSets * num_input_nodes);
    double* training_outputs = malloc(sizeof(double) * numTrainingSets * num_output_nodes);
        
    //double* hiddenLayerBias = malloc(sizeof(double) * num_hidden_nodes);
    //double* outputLayerBias= malloc(sizeof(double) * num_output_nodes);

    //double* hiddenWeights = malloc( sizeof(double) * num_input_nodes * num_hidden_nodes);
    //double* outputWeights = malloc(sizeof(double) * num_hidden_nodes * num_output_nodes);
   
   


    training_inputs = set_training_data(training_inputs, numTrainingSets, num_input_nodes, path_td);
    training_outputs = set_ouput_data(training_outputs, numTrainingSets, num_output_nodes, path_od);


    int correct_guesses = 0;

    // trains neural network with given images, labels, and weights
    correct_guesses = train(training_inputs, training_outputs, numTrainingSets,hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights);
    
    //printf("Correct guesses in  EXECUTE TRAINING: %d\n", correct_guess);

    free(training_inputs);
    free(training_outputs);
    
    return correct_guesses;



}



int main(int argc, const char * argv[]) {

    //test();
    
    int digits;
   
    int first_time = 1;
    int epochs = 1000;

    double* hiddenLayerBias = malloc(sizeof(double) * num_hidden_nodes);
    double* outputLayerBias= malloc(sizeof(double) * num_output_nodes);

    double* hiddenWeights = malloc( sizeof(double) * num_input_nodes * num_hidden_nodes);
    double* outputWeights = malloc(sizeof(double) * num_hidden_nodes * num_output_nodes);


    // creates weights
    for (int i=0; i<num_input_nodes; i++) {
        for (int j=0; j<num_hidden_nodes; j++) {
            hiddenWeights[i *num_hidden_nodes + j] = init_weight();
        }
    }
    for (int i=0; i<num_hidden_nodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j=0; j<num_output_nodes; j++) {
            outputWeights[i * num_output_nodes + j] = init_weight();
        }
    }
    for (int i=0; i<num_output_nodes; i++) {
        outputLayerBias[i] = init_weight();
    }

    save_weights(num_input_nodes, num_hidden_nodes, hiddenWeights,"W_Data/weightsHidden.dat");
    save_weights(num_hidden_nodes, num_output_nodes, outputWeights, "W_Data/weightsOutput.dat");
    save_bias(num_hidden_nodes, hiddenLayerBias,"W_Data/bias_hidden.dat" );
    save_bias(num_output_nodes, outputLayerBias,"W_Data/bias_output.dat" );

    
    int num_training_sets = 0;
    int incorrect = 0;
    float acc  = 0;

    for (int e = 0; e < epochs; e++) {
        //sets correct to 0 before every epoc
        printf("\nEpoch #%d starting...\n", e + 1);
        int correct = 0;
        
        int incorrect = 0;
        //for (int batch = 0; batch <= amount_of_batches; batch++) {
        for (int batch = 0; batch <= 0; batch++) {
            int correct_in_batch = 0;
            int batch_size = 32;
            if (batch == 186) {
                batch_size = 15;
            }
            digits = 2;
            if (batch/10 <= 0) digits = 1;

            char batch_path[15 + digits];
            char label_path[15 + digits];

            sprintf(batch_path, "Data/batch%d.txt",  batch);
            sprintf(label_path, "Data/label%d.txt",  batch);

           


            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ///// EXECUTE TRAINING //////////////////////////////////
            double* training_inputs = malloc(sizeof(double) * batch_size * num_input_nodes);
            double* training_outputs = malloc(sizeof(double) * batch_size * num_output_nodes);

            training_inputs = set_training_data(training_inputs, batch_size, num_input_nodes, batch_path);
            training_outputs = set_ouput_data(training_outputs, batch_size, num_output_nodes, label_path);
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            // TRAIN //////////////////////////////////////////////////////////////
            double hiddenLayer[num_hidden_nodes];
            double outputLayer[num_output_nodes];

            // array that stores the order of the batch
            int trainingSetOrder[batch_size];

            for (int k =0; k < batch_size; k++) {
                trainingSetOrder[k] = k;
            }
            
            int i;


                // shuffles order
                shuffle(trainingSetOrder,batch_size);

                // iterates through all images of the batch
                for (int x=0; x < batch_size; x++) {
                    
                    // image selected
                    i = trainingSetOrder[x];

                    
                    for (int j=0; j< num_hidden_nodes; j++) {
                        double activation = hiddenLayerBias[j];
                        for (int k=0; k < num_input_nodes; k++) {
                            activation += (PIXEL_SCALE(training_inputs[i*num_input_nodes + k])  * hiddenWeights[k * num_hidden_nodes + j]);
                        }
                        hiddenLayer[j] = sigmoid(activation);
                    }

                    //printf("\nNN OUTPUT: ");
                    for (int j=0; j<num_output_nodes; j++) {
                        double activation=outputLayerBias[j];
                        for (int k=0; k<num_hidden_nodes; k++) {
                            activation += hiddenLayer[k] * outputWeights[k * num_output_nodes + j];
                        }
                        outputLayer[j] = sigmoid(activation);
                        printf("%lf ", outputLayer[j]);
                    }


                print_network_output(outputLayer);
                print_network_output(training_outputs + (i * num_output_nodes + 0) );
                printf("\n");
                    if ((get_letter(outputLayer)) == (get_letter(training_outputs + (i * num_output_nodes + 0)) )) {
                        correct++;
                        printf("Correct\n");
                    } else {
                        incorrect++;
                    }
                
                // Backprop
                    double deltaOutput[num_output_nodes];
                    for (int j=0; j<num_output_nodes; j++) {

                        double errorOutput = (training_outputs[i * num_output_nodes + j] - outputLayer[j]);
                        deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
                    }
                    
                    double deltaHidden[num_hidden_nodes];
                    for (int j=0; j<num_hidden_nodes; j++) {
                        double errorHidden = 0.0f;
                        for(int k=0; k<num_output_nodes; k++) {
                            errorHidden += deltaOutput[k] * outputWeights[j * num_output_nodes +k];
                        }
                        deltaHidden[j] = errorHidden * dSigmoid(hiddenLayer[j]);
                    }
                    
                    for (int j=0; j<num_output_nodes; j++) {
                        outputLayerBias[j] += deltaOutput[j] * learning_rate;
                        for (int k=0; k<num_hidden_nodes; k++) {
                            outputWeights[k * num_output_nodes + j] += hiddenLayer[k] * deltaOutput[j] * learning_rate;
                        }
                    }
                    
                    for (int j=0; j<num_hidden_nodes; j++) {
                        hiddenLayerBias[j] += deltaHidden[j]*learning_rate;
                        for(int k=0; k<num_input_nodes; k++) {
                            //*(training_inputs +i*num_input_nodes + k)
                            hiddenWeights[k *num_hidden_nodes + j] += training_inputs[i*num_input_nodes + k] * deltaHidden[j]*learning_rate;
                        }
                    }
                }

            
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////




            free(training_inputs);
            free(training_outputs);
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////
            ///// EXECUTE TRAINING //////////////////////////////////
                        
        }
        // stats after a whole epoc
        printf("Correct Guesses: %d\n", correct);
        printf("Incorrect Guesses: %d\n", incorrect);
        acc = (float)  ((float) correct / ((float) 32));
        printf("Accuracy:  %f\n",  acc);



    }
    // frees the stack being used
    free(hiddenLayerBias);
    free(outputLayerBias);
    free(hiddenWeights);
    free(outputWeights); 

    return 0;


}