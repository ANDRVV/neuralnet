#pragma once

// -- [GLOBAL LIBRARIES] --
#include <iostream>
#include <vector>

// -- [LOCAL LIBRARIES] --
#include "mathfuncs.hpp"

struct LayerLayout final {
    int qtyInputs;
    std::vector<int> qtyHiddenLayers; // Length is quantity of hidden layers, the content is the number of neurons per layer
    int qtyOutputs;
};

class Neuron final {
    public:
        std::vector<double> weights;
        double predicted;
        double bias;

        Neuron() noexcept = default;

        explicit Neuron(int n_inputs) noexcept {
            weights.resize(n_inputs);
        }

        [[gnu::hot]] void
        predict(std::vector<double> inputs) noexcept {
            double preprocessed = 0.0;
            for (std::size_t i = 0; i < inputs.size(); i++) {
                preprocessed += inputs[i] * weights[i];
            }
            predicted = swish(preprocessed + bias);
        }
};

// Local struct. Please don't use it.
struct NeuronsLayers final {
    std::vector<std::vector<Neuron>> hiddenLayers; // 2-Dimensional (Neurons)
    std::vector<Neuron> outputLayer; // Output (Neurons)
};

// Initialize neurons with He function to make layers.
class DeepLayers final {
    private:
        _HIGH_PERF Neuron
        initNeuron(int prevNeurons) noexcept {
            Neuron n(prevNeurons);
            const double HeCode = genHe(prevNeurons);
            for (int i = 0; i < prevNeurons; i++) {
                n.weights[i] = HeCode;
            }
            // Bias to 0.0 is enough
            return n;
        }

    public:
        NeuronsLayers NetworkLayers;
        std::size_t TotalLogicLayers;
        std::size_t TotalNeurons; // Input are NOT neurons

        explicit DeepLayers(LayerLayout layout) noexcept {
            if (layout.qtyHiddenLayers.empty()) {
                NetworkLayers.outputLayer.resize(layout.qtyOutputs, initNeuron(static_cast<int>(layout.qtyInputs)));
                TotalLogicLayers = 1;
                TotalNeurons = layout.qtyOutputs;
            } else {
                std::size_t TotalHiddenNeurons = 0;

                NetworkLayers.hiddenLayers.resize(layout.qtyHiddenLayers.size());

                // Initialize FIRST layer with numbers of input (to gen He)
                NetworkLayers.hiddenLayers[0].resize(layout.qtyHiddenLayers[0], initNeuron(layout.qtyInputs));
                TotalHiddenNeurons += static_cast<std::size_t>(layout.qtyHiddenLayers[0]);
                for (std::size_t i = 1; i < layout.qtyHiddenLayers.size(); i++) {
                    int qtyNeurons = layout.qtyHiddenLayers[i];
                    TotalHiddenNeurons += static_cast<std::size_t>(qtyNeurons);

                    // Initialize layer with previous layer size (to gen He)
                    NetworkLayers.hiddenLayers[i].resize(qtyNeurons, initNeuron(static_cast<int>(NetworkLayers.hiddenLayers[i-1].size())));
                }
                
                NetworkLayers.outputLayer.resize(layout.qtyOutputs, initNeuron(static_cast<int>(NetworkLayers.hiddenLayers.back().size())));

                TotalLogicLayers = layout.qtyHiddenLayers.size() + 1;
                TotalNeurons = TotalHiddenNeurons + layout.qtyOutputs;
            }
        }
};