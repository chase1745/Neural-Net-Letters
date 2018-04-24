// Chase McDermott
// 524004424
// CSCE 420
// Due: April 23, 2018
// Neural_Net.h

#include <vector>
#include <map>

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

struct Example {
    std::vector<int> input;  // input
    std::vector<double> output; // output
};

struct Layer {
    int size;
    std::vector<std::vector<double>> weights;  // weights of each neuron
};

class Neural_Net {

private:
    int num_layers;  // not including input layer
    double learning_variable;
    std::vector<Layer> layers; // vector of Layer structs
    double rand_weight_lower;
    double rand_weight_upper;

    double activation_func(double x, double l, bool prime);
    double log_sigmoid(double x);
    double log_sigmoid_prime(double x);
    double h_tan_sigmoid(double x);
    double h_tan_sigmoid_prime(double x);
    double dot_product(std::vector<double> a, int l, int j, bool i_first);
    std::vector<double> get_random_doubles(int size, double lower_bound, double upper_bound);
    void initialize_random_weights();
    double test_input(std::vector<int> input);
public:
    static std::map<int, char> ALPHABET;

    Neural_Net(std::vector<int> sizes, double a, double rand_weight_low, double rand_weight_high);
    void back_prop_learning(std::vector<Example> examples, std::vector<Example> testing_examples, int max_epochs);
    int test_single_input(std::vector<int> input);
    double test_multiple_inputs(std::vector<Example> examples);
};


#endif //NEURAL_NET_H
