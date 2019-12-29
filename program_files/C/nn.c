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


/**
 * Given an output of num_output_nodes
 * Prints letter given by the Neural Network
 */ 
void get_network_output(double* output) {
    int letter_pos;
    double highest = 0;


    for (int i =0; i < num_output_nodes; i ++) {
        printf("Output is: %f, Highest is: %f\n", output[i], highest);
        if (output[i] > highest) {
            letter_pos = i;
            highest = output[i];

        }
    }

    printf("Position: %d", letter_pos);
    
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


int main(int argc, const char * argv[]) {

    double outputLayer[num_output_nodes];
    outputLayer[0] = 0.3;
    outputLayer[1] = 0.1;
    outputLayer[2] = 0.33;
    outputLayer[3] = 0.2;
    outputLayer[4] = 0.6;
    outputLayer[5] = 0.1;
    outputLayer[6] = 0.1;

    get_network_output(outputLayer);
     
    
    return 0;


}