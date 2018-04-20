//
// Created by Chase McDermott on 4/13/18.
//

#include <vector>

#ifndef ANNPROJECT_NEURAL_NET_H
#define ANNPROJECT_NEURAL_NET_H

struct Example {
    std::vector<int> matrix;  // input
    std::vector<double> letters; // output
};

class Neural_Net {

private:
    int num_layers;  // not including input layer
    double learning_variable;
    std::vector<int> sizes;  // size of each layer
    std::vector<std::vector<double>> weights;  // weights of each neuron
    double rand_weight_lower;
    double rand_weight_upper;

    double activation_func(double x, double l, bool prime);
    double log_sigmoid(double x);
    double log_sigmoid_prime(double x);
    double h_tan_sigmoid(double x);
    double h_tan_sigmoid_prime(double x);
    double get_summation(int n, int j, std::vector<double> a);
    std::vector<double> get_random_doubles(int size, double lower_bound, double upper_bound);
    public:
    Neural_Net(std::vector<int> sizes, double a);
    bool back_prop_learning(std::vector<Example> examples, int epochs);
};


#endif //ANNPROJECT_NEURAL_NET_H
