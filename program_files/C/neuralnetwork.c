//
//  
//  NeuralNetwork
//
//  Created by Daniel Alvarado Bonilla
//
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
double sigmoid(double x) {
    return 1 / (1 + exp(-x)); 
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
    }

    return letter;


}



/**
 * 
 * Method used to test data
 * 
 */ 
void testing_data(double training_inputs[], double training_outputs[], int num_images, int inputSize, int outputSize) {

    
    
    int num_input_nodes = 784;
    int num_output_nodes = 5;


    double hiddenLayer[num_hidden_nodes];
    double* outputLayer= malloc(sizeof(double) * num_output_nodes) ; //[num_output_nodes];
    
    double* hiddenLayerBias = malloc(sizeof(double) * num_hidden_nodes);
    double* outputLayerBias= malloc(sizeof(double) * num_output_nodes);

    // hiddenWeights[num_input_nodes][num_hidden_nodes]
    double* hiddenWeights = malloc( sizeof(double) * num_input_nodes * num_hidden_nodes);
    //double outputWeights[num_hidden_nodes][num_output_nodes];
    double* outputWeights = malloc(sizeof(double) * num_hidden_nodes * num_output_nodes);
   
    hiddenWeights = loadWeights(num_input_nodes, num_hidden_nodes,"W_Data/weightsHidden.dat" , hiddenWeights);
    outputWeights = loadWeights(num_hidden_nodes, num_output_nodes, "W_Data/weightsHidden.dat", outputWeights);
    hiddenLayerBias = loadBias(num_hidden_nodes, "W_Data/bias_hidden.dat");
    outputLayerBias = loadBias(num_output_nodes,"W_Data/bias_output.dat");

    
    int i;

    int correct = 0;
    int incorrect = 0;

        for (int i=0; i < num_images; i++) {
            
            // Forward pass (only - testing)
            
            for (int j=0; j < num_hidden_nodes; j++) {
                double activation=hiddenLayerBias[j];
                 for (int k=0; k<num_input_nodes; k++) {
                    activation+= ( PIXEL_SCALE(*(training_inputs +i*num_input_nodes + k))  * hiddenWeights[k *num_hidden_nodes + j]);
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            
            for (int j=0; j<num_output_nodes; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k < num_hidden_nodes; k++) {
                    activation += hiddenLayer[k]*outputWeights[k * num_output_nodes + j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            printf("Label: %s", testing_get_letter(training_outputs +i*num_output_nodes));
            printf("Label: %s", testing_get_letter(outputLayer));
            


            
        }

    

    free(hiddenLayerBias);
    free(outputLayerBias);
    free(hiddenWeights);
    free(outputWeights);

    }


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
int trainData(double training_inputs[], double training_outputs[], int numTrainingSets, int batch, int epocs ) 
    
    {
    
    int num_input_nodes = 784;
    int num_output_nodes = 5;


    double hiddenLayer[num_hidden_nodes];
    double* outputLayer= malloc(sizeof(double) * num_output_nodes) ; //[num_output_nodes];
    
    double* hiddenLayerBias = malloc(sizeof(double) * num_hidden_nodes);
    double* outputLayerBias= malloc(sizeof(double) * num_output_nodes);

    double* hiddenWeights = malloc( sizeof(double) * num_input_nodes * num_hidden_nodes);\
    double* outputWeights = malloc(sizeof(double) * num_hidden_nodes * num_output_nodes);
   
   // Sets randoms if flag
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
       
       hiddenWeights = loadWeights(num_input_nodes, num_hidden_nodes,"W_Data/weightsHidden.dat" , hiddenWeights);
       outputWeights = loadWeights(num_hidden_nodes, num_output_nodes, "W_Data/weightsHidden.dat", outputWeights);
       hiddenLayerBias = loadBias(num_hidden_nodes, "W_Data/bias_hidden.dat");
       outputLayerBias = loadBias(num_output_nodes,"W_Data/bias_output.dat");

   }

    

    int trainingSetOrder[numTrainingSets];

    for (int k =0; k < numTrainingSets; k++) {
        trainingSetOrder[k] = k;
    }
    
    int i;

    int correct = 0;
    int incorrect = 0;

    printf("\nBATCH %d\n", batch);

    for (int n=0; n < epocs; n++) {
        shuffle(trainingSetOrder,numTrainingSets);
        for (int x=0; x<numTrainingSets; x++) {
            
            i = trainingSetOrder[x];

            
            for (int j=0; j<num_hidden_nodes; j++) {
                double activation=hiddenLayerBias[j];
                 for (int k=0; k<num_input_nodes; k++) {
                    activation+= ( PIXEL_SCALE(*(training_inputs +i*num_input_nodes + k))  * hiddenWeights[k *num_hidden_nodes + j]);
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            
            for (int j=0; j<num_output_nodes; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<num_hidden_nodes; k++) {
                    activation += hiddenLayer[k]*outputWeights[k * num_output_nodes + j];
                }
                outputLayer[j] = sigmoid(activation);
            }


            if (correct_guess(outputLayer, training_outputs, i)) {
                correct++;
            } else {
                incorrect++;
            }

            /*
           if ((n == 0 ) || (n == epocs/2) || (n == epocs - 1) ) {
                printf("Epoc:%d ", n);
                printf("Batch:%d\n", batch);
                printf("         Output:%lf,%lf,%lf,%lf,%lf\n", outputLayer[0], outputLayer[1], outputLayer[2], outputLayer[3], outputLayer[4]);
                printf("Expected Output:%lf,%lf,%lf,%lf,%lf\n", 
                training_outputs[i * num_output_nodes + 0], training_outputs[i * num_output_nodes + 1], training_outputs[i * num_output_nodes + 2],
                training_outputs[i * num_output_nodes + 3],training_outputs[i * num_output_nodes + 4]);
                }
            //*/
           
           // Backprop
        
    
            double deltaOutput[num_output_nodes];
            for (int j=0; j<num_output_nodes; j++) {
                double errorOutput = (*(training_outputs +i*num_output_nodes + j) -outputLayer[j]);
                deltaOutput[j] = errorOutput*dSigmoid(outputLayer[j]);
            }
            
            double deltaHidden[num_hidden_nodes];
            for (int j=0; j<num_hidden_nodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<num_output_nodes; k++) {
                    errorHidden += deltaOutput[k] * outputWeights[j * num_output_nodes +k];
                }
                deltaHidden[j] = errorHidden*dSigmoid(hiddenLayer[j]);
            }
            
            for (int j=0; j<num_output_nodes; j++) {
                outputLayerBias[j] += deltaOutput[j]*lr;
                for (int k=0; k<num_hidden_nodes; k++) {
                    outputWeights[k * num_output_nodes + j]+=hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }
            
            for (int j=0; j<num_hidden_nodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j]*lr;
                for(int k=0; k<num_input_nodes; k++) {
                    hiddenWeights[k *num_hidden_nodes + j] += *(training_inputs +i*num_input_nodes + k) * deltaHidden[j]*lr;
                }
            }
            //*/
        }

    }

    // saves weights
    saveWeights(num_input_nodes, num_hidden_nodes, hiddenWeights,"W_Data/weightsHidden.dat");
    saveWeights(num_hidden_nodes, num_output_nodes, outputWeights, "W_Data/weightsOutput.dat");
    saveBias(num_hidden_nodes, hiddenLayerBias,"W_Data/bias_hidden.dat" );
    saveBias(num_output_nodes, outputLayerBias,"W_Data/bias_output.dat" );

    free(hiddenLayerBias);
    free(outputLayerBias);
    free(hiddenWeights);
    free(outputWeights);

    
    printf("Correct Guesses: %d\n", correct);
    printf("Incorrect Guesses: %d\n", incorrect);
    float acc = (float)  ((float) correct / ((float) numTrainingSets * (float) epocs));
    printf("Accuracy:  %f\n",  acc);



}

int trainData_2(double training_inputs[], double training_outputs[], int numTrainingSets, int batch, int epocs ) 
    
    {
    
    int num_input_nodes = 784;
    int num_output_nodes = 5;


    double hiddenLayer[num_hidden_nodes];
    double* outputLayer= malloc(sizeof(double) * num_output_nodes) ; //[num_output_nodes];
    
    double* hiddenLayerBias = malloc(sizeof(double) * num_hidden_nodes);
    double* outputLayerBias= malloc(sizeof(double) * num_output_nodes);

    double* hiddenWeights = malloc( sizeof(double) * num_input_nodes * num_hidden_nodes);\
    double* outputWeights = malloc(sizeof(double) * num_hidden_nodes * num_output_nodes);
   
   // Sets randoms if flag
   if (1 == 0) {
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
       
       hiddenWeights = loadWeights(num_input_nodes, num_hidden_nodes,"W_Data/weightsHidden.dat" , hiddenWeights);
       outputWeights = loadWeights(num_hidden_nodes, num_output_nodes, "W_Data/weightsHidden.dat", outputWeights);
       hiddenLayerBias = loadBias(num_hidden_nodes, "W_Data/bias_hidden.dat");
       outputLayerBias = loadBias(num_output_nodes,"W_Data/bias_output.dat");

   }

    

    int trainingSetOrder[numTrainingSets];

    for (int k =0; k < numTrainingSets; k++) {
        trainingSetOrder[k] = k;
    }
    
    int i;

    int correct = 0;
    int incorrect = 0;

    printf("\nBATCH %d\n", batch);

    for (int n=0; n < epocs; n++) {
        shuffle(trainingSetOrder,numTrainingSets);
        for (int x=0; x<numTrainingSets; x++) {
            
            i = trainingSetOrder[x];

            
            for (int j=0; j<num_hidden_nodes; j++) {
                double activation=hiddenLayerBias[j];
                 for (int k=0; k<num_input_nodes; k++) {
                    activation+= ( PIXEL_SCALE(*(training_inputs +i*num_input_nodes + k))  * hiddenWeights[k *num_hidden_nodes + j]);
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            
            for (int j=0; j<num_output_nodes; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<num_hidden_nodes; k++) {
                    activation += hiddenLayer[k]*outputWeights[k * num_output_nodes + j];
                }
                outputLayer[j] = sigmoid(activation);
            }


            if (correct_guess(outputLayer, training_outputs, i)) {
                correct++;
            } else {
                incorrect++;
            }
            printf("Label: %s", testing_get_letter(training_outputs +i*num_output_nodes));
            printf("Label: %s", testing_get_letter(outputLayer));

        }

    }

    // saves weights
    saveWeights(num_input_nodes, num_hidden_nodes, hiddenWeights,"W_Data/weightsHidden.dat");
    saveWeights(num_hidden_nodes, num_output_nodes, outputWeights, "W_Data/weightsOutput.dat");
    saveBias(num_hidden_nodes, hiddenLayerBias,"W_Data/bias_hidden.dat" );
    saveBias(num_output_nodes, outputLayerBias,"W_Data/bias_output.dat" );

    free(hiddenLayerBias);
    free(outputLayerBias);
    free(hiddenWeights);
    free(outputWeights);

    
    printf("Correct Guesses: %d\n", correct);
    printf("Incorrect Guesses: %d\n", incorrect);
    float acc = (float)  ((float) correct / ((float) numTrainingSets * (float) epocs));
    printf("Accuracy:  %f\n",  acc);



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

    //printf("THE NUMBER SCANNED IS:%lf\n", training_outputs[1]);

    fclose(file);

    return training_outputs;
}


/**
 *  Method called by main function
 *  amount refers to the amount of epochs
 *  first time refers to the rand_flag used in trainData()
 *  paths refer to the batch paths
 * 
 */ 
void executeNNtesting(int amount, int numTrainingSets) {
    // amount of data
    //int numTrainingSets = 32;
    int inputSize = 784;
    int outputSize = 5;

    char path_td[] = "Data/batch00.text";
    char path_od[] = "Data/label00.text";
    

    double* training_inputs = malloc(sizeof(double) * numTrainingSets * inputSize);
    double* training_outputs = malloc(sizeof(double) * numTrainingSets * outputSize);

    training_inputs = setTrainingData(training_inputs, numTrainingSets, inputSize, path_td);
    training_outputs = setOuputData(training_outputs, numTrainingSets, outputSize, path_od);
    
    trainData_2(training_inputs, training_outputs, numTrainingSets, 1, amount);
    
    
    free(training_inputs);
    free(training_outputs);

}

void executeNNtraining(int amount, int numTrainingSets ,int first_time, char path_td[]
            ,char path_od[]) {
    // amount of data
    //int numTrainingSets = 32;
    int inputSize = 784;
    int outputSize = 5;


    double* training_inputs = malloc(sizeof(double) * numTrainingSets * inputSize);
    double* training_outputs = malloc(sizeof(double) * numTrainingSets * outputSize);

    training_inputs = setTrainingData(training_inputs, numTrainingSets, inputSize, path_td);
    training_outputs = setOuputData(training_outputs, numTrainingSets, outputSize, path_od);
    
    trainData(training_inputs, training_outputs, numTrainingSets, first_time, amount);
    
    free(training_inputs);
    free(training_outputs);

}






/**
 * Method used to test different functions
 * 
 */ 
void test() {


    int numTrainingSets = 32;
    int inputSize = 784;
    int outputSize = 5;


    double* training_inputs = malloc(sizeof(double) * numTrainingSets * inputSize);
    //double* training_outputs = malloc(sizeof(numTrainingSets * outputSize));

    training_inputs = setTrainingData(training_inputs, numTrainingSets, inputSize, "Data/batch60.txt");
    //training_outputs = setOuputData(training_outputs, numTrainingSets, outputSize, "label0.txt");
    
    //trainData(training_inputs, training_outputs, numTrainingSets, first_time, amount);
    
    free(training_inputs);
    //free(training_outputs);

}




int main(int argc, const char * argv[]) {

    //test();
    
    int digits;
   
    int first_time = 1;
    int epochs = 1000;

    ///*
    for (int i = 0; i <= 79; i++) {
        int batch_size = 32;
        if (i == 79) {
            batch_size = 2;
        }
        digits = 2;
        if (i/10 <= 0) digits = 1;

        char batch_path[15 + digits];
        char label_path[15 + digits];

        sprintf(batch_path, "Data/batch%d.txt",  i);
        sprintf(label_path, "Data/label%d.txt",  i);

        //printf("%s\n",batch_path);
        //printf("%s\n",label_path);
        //executeNNtesting()    

        //executeNNtraining(100, batch_size,i,batch_path, label_path);
        executeNNtesting(1, 3); 
    }
    //*/

    return 0;


}