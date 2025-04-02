#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include "src/neuralnet.hpp"

void getPercs(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double value : vec) {
        if (value != 0) {
            sum += value;
        }
    }
    std::vector<std::pair<int, double>> indexToPercentage;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        double value = vec[i];
        if (value >= 0.01 && sum > 0) {
            double percentage = (value / sum) * 100.0;
            indexToPercentage.emplace_back(i, percentage);
        }
    }
    std::sort(indexToPercentage.begin(), indexToPercentage.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });

    for (std::pair<int, double> entry : indexToPercentage) {
        std::printf("%d: %.f%%, ", entry.first, entry.second);
    }
    std::printf("\n-------------\n");
}

int main() {
    LayerLayout layout;
    layout.qtyInputs = 23;
    layout.qtyOutputs = 10;
    layout.qtyHiddenLayers = {18};

    DeepLayers dl(layout); // Initialize the layers

    NeuralNetwork::DeepNetwork DDN(dl);

    std::vector<NeuralNetwork::Dataset> traindata = {
        { .input = {1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1}, .target = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0} },
        { .input = {0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1}, .target = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0} },
        { .input = {1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1}, .target = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0} },
        { .input = {1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1}, .target = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0} },
        { .input = {1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1}, .target = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0} },
        { .input = {1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1}, .target = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0} },
        { .input = {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1}, .target = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0} },
        { .input = {1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1}, .target = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0} },
        { .input = {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1}, .target = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0} },
        { .input = {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1}, .target = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1} }
    };

    DDN.Train(traindata, 100000);
    
    for (const NeuralNetwork::Dataset& data : traindata) {
        std::vector<double> res = DDN.GetOutput(data.input);
        //getPercs(res);
    }
    std::vector<double> res = DDN.GetOutput(std::vector<double>{
        1, 1, 1,
        1, 0, 1, 
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
            
             });
    getPercs(res);

    for (double r : res) {
        std::cout << r << std::endl;
    }

    return 0;
}