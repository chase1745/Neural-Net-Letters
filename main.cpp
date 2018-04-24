// Chase McDermott
// 524004424
// CSCE 420
// Due: April 23, 2018
// main.cpp

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <bitset>
#include "Neural_Net.h"

using namespace std;

string hex_to_binary(const string &hex_string) {
    stringstream ss;
    ss << hex << hex_string;
    unsigned n;
    ss >> n;
    bitset<7> b(n);
    return b.to_string();
}

vector<int> b_string_to_vector(const string &binary_str) {
    std::vector<int> binary_vec;
    for(auto str : binary_str) {
        binary_vec.push_back(str - '0'); // Convert '1' or '0' to int.
    }
    return binary_vec;

}

vector<Example> get_letters(const string &file_name) {
    vector<Example> examples;
    vector<int> binary;
    string hex;
    int i = 0;
    int j = 0;
    ifstream inFile(file_name);

    if(inFile.is_open()) {

        while(inFile >> hex) {
            if(i > 4) {
                // End of letter.
                Example example;
                example.output = vector<double>(26, 0);
                example.output[j] = 1;
                example.input = binary;
                examples.push_back(example);

                // Reset everything.
                j++;
                i = 0;
                binary.clear();
            }
            vector<int> new_vect = b_string_to_vector(hex_to_binary(hex));
            binary.insert(binary.end(), new_vect.begin(), new_vect.end());  // Concatenate the vectors.
            i++;
        }
        inFile.close();

        // Get last letter
        Example example;
        example.output = vector<double>(26, 0);
        example.output[j] = 1;
        example.input = binary;
        examples.push_back(example);

    }
    else cerr << "Unable to open file";
    return examples;
}

Example random_bit_flipped_example(int num_bits, Example correct_example) {
    srand(time(NULL));
    Example example = correct_example;
    vector<double> correct_input = correct_example.output;
    for(int i = 0; i < num_bits; i++) {
        int random_num = rand() % 35;
        (correct_example.input[random_num] == 1) ? example.input[random_num] = 0 : example.input[random_num] = 1;
    }
    return example;
}

vector<int> max_random_bit_flips(vector<Example> examples, Neural_Net network) {
    vector<int> max_bits_flipped;
    int num_bits;

    for(int i = 0; i < 26; i++) {
        // each letter
        num_bits = 1;
        while(true) {
            Example example = random_bit_flipped_example(num_bits, examples[i]);
            if(network.test_single_input(example.input) != i)
                break;
            else
                num_bits++;
        }
        max_bits_flipped.push_back(num_bits);
    }
    return max_bits_flipped;
}

vector<double> average_max_bit_flips(int num_iterations, vector<Example> examples, Neural_Net network) {

    vector<vector<int>> ave_maxes;
    vector<double> aves;
    for(int i = 0; i < num_iterations; i++) {
        ave_maxes.push_back(max_random_bit_flips(examples, network));
    }
    for(int i = 0; i < 26; i++) {
        double sum = 0;
        for(int j = 0; j < num_iterations; j++) {
            sum += ave_maxes[j][i];
        }
        aves.push_back(sum/num_iterations);
    }
    return aves;
}

void user_input(Neural_Net network) {
    vector<string> raw_inputs;
    vector<int> inputs;
    string input;
    char single_input;
    while(true) {
        cout << "Please enter a 35 bit input representing a 5x7 binary letter (ending with a newline)." << endl;
        cin.clear();
        cin.ignore(INT8_MAX);
        getline(cin, input);
        for(int i = 0; i < input.size(); i++) {
            if(input[i] != ' ')
                inputs.push_back((input[i] - '0'));
        }
        if (inputs.size() != 35) {
            cout << endl << "Your input was not 35 bits long (" << inputs.size();
            cout << ") would you like to try again? (Y for yes, N for no) ";
            cin >> single_input;
            if(single_input == 'Y' || single_input == 'y')
                continue;
            else
                break;
        } else {
            int output_char = network.test_single_input(inputs);
            cout << "The Artificial Neural Network thinks your input represents a(n): " << network.ALPHABET[output_char] << endl;
        }
        cout << "Would you like to give another input? (Y for yes, N for no) " << endl;
        cin >> single_input;
        if(single_input == 'Y' || single_input == 'y')
            continue;
        else
            break;
    }
}

int main() {
    vector<Example> examples = get_letters("examples.txt");
    vector<Example> testing_examples = examples;
    testing_examples.insert(testing_examples.begin(), examples.begin(), examples.begin());

    vector<int> sizes{ 35, 35, 26 };
    Neural_Net network = Neural_Net(sizes, .15, -0.1, 0.1);
    network.back_prop_learning(examples, testing_examples, 1000);

    double accuracy = network.test_multiple_inputs(examples);
    cout << "Average accuracy with correct inputs: " << accuracy*100 << "%" << endl << endl;

    vector<double> max_aves = average_max_bit_flips(100, examples, network);
    for(int i = 0; i < 26; i++) {
        cout << "Average max bits flipped through 100 trials for ";
        cout << network.ALPHABET[i] << ": " << max_aves[i] << endl;
    }
    cout << endl;

    user_input(network);


    return 0;
}