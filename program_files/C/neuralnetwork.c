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

double* loadWeights(int rows, int cols, char path[], double* weights) {
    
    FILE *file;
    file = fopen(path, "r");
    int i, j;

    fread(&weights[0],sizeof(double), rows * cols , file);

    fclose(file);

    printf("Weight 1: %lf", weights[0]);
    printf("Weight 2: %lf", weights[1]);
    


    return weights;
}




/* Method used to save weights into a simple dat file
 * Not optimized
 * 
 * "ImageCrop/weightsHidden.dat"
 */

int saveWeights(int rows, int cols, double* weights, char path[]) {
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

double* loadBias(int size, char path[]) {
    
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

int saveBias(int size, double* bias, char path[]) {
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

   w = loadWeights(r, c, path, w);

   

}

/**
 * Given an output of num_output_nodes
 * Returns the Letter with the highest output 
 */ 
char* testing_get_letter(double* output) {
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
                double* hiddenWeights, double* outputWeights,
                int batch, int epocs ) 
    
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

    printf("\nBATCH %d\n", batch);


    //sets weights
    
    // sets weights for each batch
    if (batch == 0) {
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

   
   } else {
       printf("Wights loaded");
       hiddenWeights = loadWeights(num_input_nodes, num_hidden_nodes,"W_Data/weightsHidden.dat" , hiddenWeights);
       outputWeights = loadWeights(num_hidden_nodes, num_output_nodes, "W_Data/weightsHidden.dat", outputWeights);
       hiddenLayerBias = loadBias(num_hidden_nodes, "W_Data/bias_hidden.dat");
       outputLayerBias = loadBias(num_output_nodes,"W_Data/bias_output.dat");

   }


    for (int n=0; n < epocs; n++) {

        // shuffles order
        shuffle(trainingSetOrder,numTrainingSets);

        // iterates through all images of the batch
        for (int x=0; x<numTrainingSets; x++) {
            
            // image selected
            i = trainingSetOrder[x];

            
            for (int j=0; j< num_hidden_nodes; j++) {
                double activation = hiddenLayerBias[j];
                 for (int k=0; k < num_input_nodes; k++) {
                    activation += (PIXEL_SCALE(*(training_inputs +i*num_input_nodes + k))  * hiddenWeights[k * num_hidden_nodes + j]);
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            printf("Hidden Weight 0: %lf\n", hiddenWeights[0]);
            printf("Hidden Weight 1: %lf\n", hiddenWeights[1]);
            printf("Hidden Weight 2: %lf\n", hiddenWeights[2]);
            printf("Hidden Weight 3: %lf\n", hiddenWeights[3]);

            for (int j=0; j<num_output_nodes; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<num_hidden_nodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k * num_output_nodes + j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            printf("Output Weight 0: %lf\n\n", outputWeights[0]);
            

            if (correct_guess(outputLayer, training_outputs, i)) {
                correct++;
            } else {
                incorrect++;
            }
           
           // Backprop
        
    
            double deltaOutput[num_output_nodes];
            for (int j=0; j<num_output_nodes; j++) {

                double errorOutput = (*(training_outputs +i * num_output_nodes + j) - outputLayer[j]);
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
                    //outputWeights[k * numOutputs + j]+=hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }
            
            for (int j=0; j<num_hidden_nodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j]*learning_rate;
                for(int k=0; k<num_input_nodes; k++) {
                    hiddenWeights[k *num_hidden_nodes + j] += *(training_inputs +i*num_input_nodes + k) * deltaHidden[j]*learning_rate;
                    //hiddenWeights[k *numHiddenNodes + j] += *(training_inputs +i*numInputs + k) * deltaHidden[j]*lr;
                }
            }
            //*/
        }

    }


    
    printf("Correct Guesses: %d\n", correct);
    printf("Incorrect Guesses: %d\n", incorrect);
    float acc = (float)  ((float) correct / ((float) numTrainingSets * (float) epocs));
    printf("Accuracy:  %f\n",  acc);

    // saves weights
    saveWeights(num_input_nodes, num_hidden_nodes, hiddenWeights,"W_Data/weightsHidden.dat");
    saveWeights(num_hidden_nodes, num_output_nodes, outputWeights, "W_Data/weightsOutput.dat");
    saveBias(num_hidden_nodes, hiddenLayerBias,"W_Data/bias_hidden.dat" );
    saveBias(num_output_nodes, outputLayerBias,"W_Data/bias_output.dat" );


}



/**
 * 
 *  Method used to set the training variable
 *  The path will be batch_name = "batch%d.dat" % (batch_n)
 * 
 * 
 */
double*  setTrainingData(double* training_inputs, int numTrainingSets, int inputSize, char path[]) {
    

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
double* setOuputData(double* training_outputs ,int numTrainingSets, 
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
    printf("img1: %lf, %lf, %lf, %lf, %lf, %f", training_outputs[0], training_outputs[1], training_outputs[2],
                training_outputs[3], training_outputs[4], training_outputs[5]);
    printf("\nimg2: %lf, %lf, %lf, %lf, %lf, %f", training_outputs[6], training_outputs[7], training_outputs[8],
                training_outputs[9], training_outputs[10], training_outputs[11]);
                


    return training_outputs;
}



/**
 * Method that loads the input data (images in each batch) and their respective output (labels)
 * 
 */ 
void execute_training(int amount, int numTrainingSets ,int batch_number, char path_td[], char path_od[]) {


    double* training_inputs = malloc(sizeof(double) * numTrainingSets * num_input_nodes);
    double* training_outputs = malloc(sizeof(double) * numTrainingSets * num_output_nodes);
        
    double* hiddenLayerBias = malloc(sizeof(double) * num_hidden_nodes);
    double* outputLayerBias= malloc(sizeof(double) * num_output_nodes);

    double* hiddenWeights = malloc( sizeof(double) * num_input_nodes * num_hidden_nodes);\
    double* outputWeights = malloc(sizeof(double) * num_hidden_nodes * num_output_nodes);
   
   


    training_inputs = setTrainingData(training_inputs, numTrainingSets, num_input_nodes, path_td);
    training_outputs = setOuputData(training_outputs, numTrainingSets, num_output_nodes, path_od);




    // trains neural network with given images, labels, and weights
    train(training_inputs, training_outputs, numTrainingSets,hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights , batch_number, amount);
    
    free(training_inputs);
    free(training_outputs);
    


    free(hiddenLayerBias);
    free(outputLayerBias);
    free(hiddenWeights);
    free(outputWeights);


}



int main(int argc, const char * argv[]) {

    //test();
    
    int digits;
   
    int first_time = 1;
    int epochs = 1000;

    for (int batch = 0; batch <= amount_of_batches; batch++) {
        int batch_size = 32;
        if (batch == 79) {
            batch_size = 15;
        }
        digits = 2;
        if (batch/10 <= 0) digits = 1;

        char batch_path[15 + digits];
        char label_path[15 + digits];

        sprintf(batch_path, "Data/batch%d.txt",  batch);
        sprintf(label_path, "Data/label%d.txt",  batch);

        execute_training(epochs, batch_size, batch, batch_path, label_path);
    }

    return 0;


}