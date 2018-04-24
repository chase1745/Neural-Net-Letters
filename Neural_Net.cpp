// Chase McDermott
// 524004424
// CSCE 420
// Due: April 23, 2018
// Neural_Net.cpp

#include "Neural_Net.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <map>
#include <algorithm>
#include <fstream>

using namespace std;

map<int, char> Neural_Net::ALPHABET = {
        { 0, 'A' },
        { 1, 'B' },
        { 2, 'C' },
        { 3, 'D' },
        { 4, 'E' },
        { 5, 'F' },
        { 6, 'G' },
        { 7, 'H' },
        { 8, 'I' },
        { 9, 'J' },
        { 10, 'K' },
        { 11, 'L' },
        { 12, 'M' },
        { 13, 'N' },
        { 14, 'O' },
        { 15, 'P' },
        { 16, 'Q' },
        { 17, 'R' },
        { 18, 'S' },
        { 19, 'T' },
        { 20, 'U' },
        { 21, 'V' },
        { 22, 'W' },
        { 23, 'X' },
        { 24, 'Y' },
        { 25, 'Z' }
};

Neural_Net::Neural_Net(std::vector<int> layer_sizes, double learning_variable, double rand_weight_lower, double rand_weight_upper) {
    // Constructor
    this->rand_weight_lower = rand_weight_lower;
    this->rand_weight_upper = rand_weight_upper;
    num_layers = (int)layer_sizes.size();
    this->learning_variable = learning_variable;
    for(int i = 0; i < num_layers; i++) {
        // initialize layers
        layers.push_back(Layer{layer_sizes[i], vector<vector<double>>()});
    }
}

void Neural_Net::back_prop_learning(vector<Example> examples, vector<Example> testing_examples, int max_epochs) {
    // Implements given (corrected) algorithm (fig. 18.24).

    bool printed = false;
    double accuracy = 0;
    // Initialize all vectors to input layer size which is largest layer.
    vector<vector<double>> in;
    vector<vector<double>> delta;
    vector<vector<double>> a;
    // Initialize vectors.
    for(int i = 0; i < num_layers; i++) {
        a.push_back(vector<double>(layers[i].size, 0));
        delta.push_back(vector<double>(layers[i].size, 0));
        in.push_back(vector<double>(layers[i].size, 0));
    }

    initialize_random_weights();
    int num_epochs = 0;
    for(num_epochs = 0; num_epochs < max_epochs; num_epochs++) {
        for (auto example : examples) {
            /* * * * * * * * * * * *
             * Forward propagation *
             * * * * * * * * * * * */
            for (int i = 0; i < layers[0].size; i++) {
                // iterate through input layer
                a[0][i] = example.input[i];
            }
            for (int l = 1; l < num_layers; l++) {
                // from first hidden layer to output layer
                for (int j = 0; j < layers[l].size; j++) {
                    // iterate through all neurons in layer l.
                    in[l][j] = dot_product(a[l - 1], l - 1, j, true);
                    a[l][j] = activation_func(in[l][j], l, false);
                }
            }


            /* * * * * * * * * * * * *
             *    Back propagation   *
             * * * * * * * * * * * * */
            for (int j = 0; j < layers[num_layers - 1].size; j++) {
                // Iterate through output layer
                int l = num_layers - 1;
                delta[l][j] = activation_func(in[l][j], num_layers - 1, true) * (example.output[j] - a[l][j]);
            }
            for (int l = num_layers - 2; l >= 1; l--) { // Skip output layer.
                // iterate from output layer to first hidden layer
                for (int i = 0; i < layers[l].size; i++) {
                    // iterate through the neurons in layer l.
                    delta[l][i] = activation_func(in[l][i], l, true) * dot_product(delta[l + 1], l, i, false);
                }
            }

            /* * * * * * * * * * * *
             *   Update weights    *
             * * * * * * * * * * * */
            for (int l = 0; l < num_layers - 1; l++) {
                // iterate through layers
                for (int i = 0; i < layers[l].size; i++) {
                    for (int j = 0; j < layers[l + 1].size; j++) {
                        layers[l].weights[i][j] =
                                layers[l].weights[i][j] + (learning_variable * a[l][i] * delta[l + 1][j]);
                    }
                }
            }
        }
        // Test 'testing_examples' to measure accuracy.
        accuracy = test_multiple_inputs(testing_examples);

        if(!printed)
            cout << '\r' << "Accuracy after " << num_epochs + 1 << " epochs: " << accuracy * 100 << "%";
        if (accuracy >= .9 && !printed) {
            // !printed to train to 100% or max_epochs
            cout << endl << "Accuracy has reached 90% at epoch: " << num_epochs + 1;
            cout << " with a learning rate of " << learning_variable << ".";
            cout << " Finishing training to 100% accuracy or max_epochs." << endl;
            printed = true;
        }
        if (accuracy >= 1) // stop when accuracy reaches 100%
            break;
    }
    cout << "Max accuracy reached after epoch " << num_epochs << ": " << accuracy*100 << "%." << endl;
}

vector<double> Neural_Net::get_random_doubles(int size, double lower_bound, double upper_bound) {
    /* Gets a 'size' number of uniformly distributed doubles between
     * lower_bound and upper_bound.
     */
    std::vector<double> list;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    for(int j = 0; j < size; j++) {
        double random_num = 0;
        while(random_num == 0)  // Make sure it's never 0.
            random_num = dis(gen);
        list.push_back(random_num);
    }
    return list;
}

double Neural_Net::dot_product(vector<double>a, int l, int j, bool i_first) {
    // Compute the dot product of a and weights at layer l.
    Layer *layer = &layers[l];
    double sum = 0;
    double add_val = 0;
    for(int i = 0; i < a.size(); i++) {
        add_val = i_first ? layer->weights[i][j] * a[i] : layer->weights[j][i] * a[i];
        sum += add_val;
    }
    return sum;
}

double Neural_Net::log_sigmoid(double x) {
    double val = ((x / (1 + abs(x))) + 1)/2;
    return val;
}

double Neural_Net::log_sigmoid_prime(double x) {
    double val = log_sigmoid(x) * (1 - log_sigmoid(x));
    return val;
}

double Neural_Net::h_tan_sigmoid(double x) {
    double val = x / (1 + abs(x));
    return val;
}

double Neural_Net::h_tan_sigmoid_prime(double x) {
    double val = 1 - pow(h_tan_sigmoid(x), 2);
    return val;
}

double Neural_Net::activation_func(double x, double l, bool prime) {
    // Function to determine which sigmoid function to apply depending
    // on the layer and whether or not to use the derivative.
    if(l < num_layers - 1) {
        // hidden layers - tanh
        return prime ? h_tan_sigmoid_prime(x) : h_tan_sigmoid(x);
    } else {
        // output layer - logistic sigmoid
        return prime ? log_sigmoid_prime(x) : log_sigmoid(x);
    }
}

void Neural_Net::initialize_random_weights() {
    // Gives each neuron in the net a random weight.
    for(int l = 0; l < num_layers - 1; l++) {  // No weights after output layer.
        // get random weights for each neuron
        Layer *layer = &layers[l];
        int next_layer_size = layers[l+1].size;
        for(int i = 0; i < layer->size; i++) {
            layer->weights.push_back(vector<double>());
            vector<double> rand_doubles = get_random_doubles(next_layer_size, rand_weight_lower, rand_weight_upper);
            for(auto rand : rand_doubles) {
                layer->weights[i].push_back(rand);
            }
        }
    }
}

double Neural_Net::test_input(std::vector<int> input) {
    // Given an input in the form of a 35 length vector of ints and returns which index the neural net thinks it represents.
    vector<vector<double>> in;
    vector<vector<double>> a;
    // Initialize vectors.
    for(int i = 0; i < num_layers; i++) {
        a.push_back(vector<double>(layers[i].size, 0));
        in.push_back(vector<double>(layers[i].size, 0));
    }

    for(int i = 0; i < layers[0].size; i++) {
        // iterate through input layer
        a[0][i] = input[i];
    }
    for (int l = 1; l < num_layers; l++) {
        // from first hidden layer to output layer
        for (int j = 0; j < layers[l].size; j++) {
            // iterate through all neurons in layer l.
            in[l][j] = dot_product(a[l-1], l-1, j, true);
            a[l][j] = activation_func(in[l][j], l, false);
        }
    }

    return distance(a[num_layers - 1].begin(), max_element(a[num_layers - 1].begin(), a[num_layers - 1].end()));
}

double Neural_Net::test_multiple_inputs(vector<Example> examples) {
    // Public function to test multiple inputs at a time, and returns the average accuracy over all tests.
    double sum = 0;
    double num_examples = examples.size();
    for(auto example : examples) {
        double max_letter_val =  test_input(example.input);
        if(max_letter_val == distance(example.output.begin(), max_element(example.output.begin(), example.output.end()))) {
            sum += 1;
        }
    }
    return sum > 0 ? sum/num_examples : 0.0;
}
