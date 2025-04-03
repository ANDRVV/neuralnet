## Overview
This program initializes and trains a deep neural network using predefined training data. It consists of an input layer, a hidden layer, and an output layer. The trained model is then used to make predictions.

## Code Explanation

### Initialize Network Architecture
```cpp
LayerLayout layout;
layout.qtyInputs = 23;
layout.qtyOutputs = 10;
layout.qtyHiddenLayers = {18}; // e.g. {10, 13, 18} for 3 hidden layer with 10, 13 and 18 neurons

DeepLayers dl(layout);
std::cout << dl.TotalLogicLayers; // print logical layers
std::cout << dl.TotalNeurons;     // print total neurons of network

NeuralNetwork::DeepNetwork DDN(dl);
```
The neural network is initialized with:
- **23 input neurons**
- **10 output neurons**
- **1 hidden layer with 18 neurons**

### Define Training Data
```cpp
std::vector<NeuralNetwork::Dataset> traindata = {
    { .input = {1, 1, 1, ...}, .target = {1, 0, 0, ...} },
    { .input = {0, 1, 0, ...}, .target = {0, 1, 0, ...} },
    ...
};
```
The dataset consists of multiple input vectors, each paired with a corresponding target output vector.

### Train the Network
```cpp
DDN.Train(traindata, 100000);
```
The neural network is trained using the dataset for **100,000 iterations**.

### Make Predictions
```cpp
for (const NeuralNetwork::Dataset& data : traindata) {
    std::vector<double> res = DDN.GetOutput(data.input);
}
```
The trained model generates predictions for each input in the training dataset.

### Test with a Custom Input
```cpp
std::vector<double> res = DDN.GetOutput(std::vector<double>{
    1, 1, 1,
    1, 0, 1, 
    1, 1, 1,
    1, 0, 1,
    1, 1, 1,
});
```
An independent test input is provided, and the resulting output is printed to the console.

### Output Results
```cpp
for (double r : res) {
    std::cout << r << std::endl;
}
```
The program prints the predicted output values to the console.

