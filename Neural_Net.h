//
// Created by Chase McDermott on 4/13/18.
//

#include <vector>

#ifndef ANNPROJECT_NEURAL_NET_H
#define ANNPROJECT_NEURAL_NET_H

struct Example {
    int matrix;  // input
    char letter; // output
};

class Neural_Net {

private:
    int num_layers;  // not including input layer
    std::vector<int> sizes;  // size of each layer
    std::vector<double> biases;  // biases for each layer
    std::vector<std::vector<double>> weights;  // weights of each neuron
    double rand_weight_lower;
    double rand_weight_upper;

    std::vector<double> get_random_doubles(int size, double lower_bound, double upper_bound);
public:
    Neural_Net(std::vector<int> sizes);
    bool back_prop_learning(std::vector<Example> examples);
};


#endif //ANNPROJECT_NEURAL_NET_H
