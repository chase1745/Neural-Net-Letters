#include <iostream>
#include <vector>
#include "Neural_Net.h"

using namespace std;

int main() {

    vector<int> sizes{ 35, 30, 26 };

    Neural_Net network = Neural_Net(sizes, .1);
    network.back_prop_learning(examples, 200);

    return 0;
}