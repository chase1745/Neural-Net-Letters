//
// Created by Chase McDermott on 4/13/18.
//

#include "Neural_Net.h"
#include <vector>
#include <random>

using namespace std;

Neural_Net::Neural_Net(std::vector<int> layer_sizes) {
    num_layers = (int)sizes.size();
    sizes = layer_sizes;
    biases = vector<double>(num_layers - 1, 0); // Initialize to 0 here, change in back prop.
    for(int i = 1; i < num_layers; i++) {  // Skip first layer (input layer).
        weights.push_back(vector<double>());  // Initialize to 0 here, change in back prop.
    }
}

bool Neural_Net::back_prop_learning(std::vector<Example> examples) {
    for(int i = 0; i < num_layers - 1; i++) {  // Skip input layer 0.
        // get random weights for each neuron
        vector<double> rand_doubles = get_random_doubles(sizes[i + 1], rand_weight_lower, rand_weight_upper);
        for(int j = 0; j < rand_doubles.size(); j++) {
            weights[i].push_back(rand_doubles[j]);
        }
    }

    while(true) {
        vector<int> inputs;
        for(int i = 0; i < examples.size(); i++) {
            inputs.push_back(examples[i].letter);
        }
        for(int l = num_layers; l >= 2; l--) {

        }
        if(SOME_CONDITION) break;
    }
}

vector<double> Neural_Net::get_random_doubles(int size, double lower_bound, double upper_bound) {
    /* Gets a 'size' number of uniformly distributed doubles between
     * lower_bound and upper_bound.
     * */
    vector<double> list;
    uniform_real_distribution<double> unif(lower_bound, upper_bound);
    default_random_engine re;
    for(int j = 0; j < size; j++) {
        double random_num = unif(re);
        list.push_back(random_num);
    }
    return list;
}