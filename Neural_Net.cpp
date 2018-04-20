//
// Created by Chase McDermott on 4/13/18.
//

#include "Neural_Net.h"
#include <vector>
#include <random>
#include <math.h>

using namespace std;

Neural_Net::Neural_Net(std::vector<int> layer_sizes, double a) {
    num_layers = (int)layer_sizes.size();
    learning_variable = a;
    sizes = layer_sizes;
    for(int i = 1; i < num_layers; i++) {  // Skip first layer (input layer).
        weights.push_back(vector<double>());  // Initialize to 0 here, change in back prop.
    }
}

bool Neural_Net::back_prop_learning(std::vector<Example> examples, int epochs) {
    // Initialize all vectors to input layer size which is largest layer.
    vector<double> in(sizes[0], 0);
    vector<double> delta(sizes[0], 0);
    vector<double> a(sizes[0], 0);
    Example example;

    for(int i = 0; i < num_layers - 1; i++) {  // Skip input layer 0.
        // get random weights for each neuron
        vector<double> rand_doubles = get_random_doubles(sizes[i + 1], rand_weight_lower, rand_weight_upper);
        for(int j = 0; j < rand_doubles.size(); j++) {
            weights[i].push_back(rand_doubles[j]);
        }
    }

    int num_epochs = 0;
    while(num_epochs < epochs) {
        for(int e = 0; e < examples.size(); e++) {
            // for each example
            example = examples[e];

            // Forward propagation
            for(int i = 0; i < sizes[0]; i++) {
                // iterate through input layer
                a[i] = example.matrix[i]);
            }
            for (int l = 1; l < num_layers; l++) {
                // from first hidden layer to output layer
                for (int j = 0; j < sizes[l]; j++) {
                    // iterate through all neurons in layer l.
                    in[j] = get_summation(sizes[l], j, a);
                    a[j] = activation_func(in[j], l, false);
                }
            }

            // Backward propagation
            for (int j = 0; j < sizes[num_layers - 1]; j++) {
                // Iterate through output layers
                delta[j] = activation_func(in[j], num_layers - 1, true) * (example.letters[j] - a[j]);
            }
            for(int l = num_layers - 1; l >= 1; l--) {
                // iterate from output layer to first hidden layer
                for(int i = 0; i < sizes[l]; i++) {
                    // iterate through layer l.
                    delta[i] = activation_func(in[i], l, true) * get_summation(sizes[l], i, delta);
                }
            }

            // Update weights
            for(int l = 0; l < num_layers; l++) {
                // iterate through layers
                for(int i = 0; i < sizes[l]; i++) {
                    // iterate through weights at each layer
                    weights[l][i] = weights[l][i] + (learning_variable * a[l] * delta[i]);
                }
            }
        }

        num_epochs++;
    }

}

vector<double> Neural_Net::get_random_doubles(int size, double lower_bound, double upper_bound) {
    /* Gets a 'size' number of uniformly distributed doubles between
     * lower_bound and upper_bound.
     * */
    std::vector<double> list;
    uniform_real_distribution<double> unif(lower_bound, upper_bound);
    default_random_engine re;
    for(int j = 0; j < size; j++) {
        double random_num = unif(re);
        list.push_back(random_num);
    }
    return list;
}

double Neural_Net::get_summation(int n, int j, vector<double>a) {
    double sum = 0;
    for(int i = 0; i < n; i++) {
        sum += weights[i][j] * a[i];
    }
    return sum;
}

double Neural_Net::log_sigmoid(double x) {
    return log(x);
}

double Neural_Net::log_sigmoid_prime(double x) {
    return log_sigmoid(x) * (1 - log_sigmoid(x));
}

double Neural_Net::h_tan_sigmoid(double x) {
    return tanh(x);
}

double Neural_Net::h_tan_sigmoid_prime(double x) {
    return 1 - pow(h_tan_sigmoid(x), 2);
}

double Neural_Net::activation_func(double x, double l, bool prime) {
    // Function to determine which sigmoid function to apply depending
    // on the layer and whether or not to use the derivative.
    if(l == num_layers - 1) {
        // output layer - logistic sigmoid
        return prime ? log_sigmoid_prime(x) : log_sigmoid(x);
    } else {
        // hidden layers - tanh
        return prime ? h_tan_sigmoid_prime(x) : h_tan_sigmoid(x);
    }
}
