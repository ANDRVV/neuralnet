#pragma once

// -- [GLOBAL LIBRARIES] --
#include <iostream>
#include <vector>

// -- [LOCAL LIBRARIES] --
#include "mathfuncs.hpp"
#include "neurons.hpp"

namespace NeuralNetwork {
    const double DEFAULT_LEARNRATE = 0.03;
    const double GRADIENT_CLIP_THRESHOLD = 10.0;

    struct Dataset {
        const std::vector<double> input;
        const std::vector<double> target;
    };

    class DeepNetwork final {
        private:
            double learnRate = DEFAULT_LEARNRATE;
            DeepLayers layout;

            _HIGH_PERF std::vector<double>
            _get_predicts(const std::vector<Neuron>& layer) const noexcept {
                std::vector<double> predicts;
                for (const Neuron& neuronnode : layer) {
                    predicts.push_back(neuronnode.predicted);
                }
                return predicts;
            }

            _HIGH_PERF_VOID void
            forward(const std::vector<double>& inputset) noexcept {
                if (layout.NetworkLayers.hiddenLayers.empty()) {
                    // input to output
                    for (std::size_t i = 0; i < layout.NetworkLayers.outputLayer.size(); i++) {
                        layout.NetworkLayers.outputLayer[i].predict(inputset);
                    }
                } else {
                    // input
                    for (std::size_t i = 0; i < layout.NetworkLayers.hiddenLayers[0].size(); i++) {
                        layout.NetworkLayers.hiddenLayers[0][i].predict(inputset);
                    }

                    // among hidden layers (start from 1)
                    for (std::size_t i = 1; i < layout.NetworkLayers.hiddenLayers.size(); i++) {
                        for (std::size_t j = 0; j < layout.NetworkLayers.hiddenLayers[i].size(); j++) {
                            layout.NetworkLayers.hiddenLayers[i][j].predict(
                                _get_predicts(layout.NetworkLayers.hiddenLayers[i-1])
                            );
                        }
                    }

                    // output neurons
                    for (std::size_t i = 0; i < layout.NetworkLayers.outputLayer.size(); i++) {
                        layout.NetworkLayers.outputLayer[i].predict(
                            _get_predicts(layout.NetworkLayers.hiddenLayers.back())
                        );
                    }
                }
            }

            _HIGH_PERF_VOID void
            _update_wb(Neuron& neuronn, const std::vector<double>& prevlayer_predicts, const double& delta) {
                double g = learnRate * delta;
                // gradient clipping to prevent gradient explosion
                if (std::abs(g) > GRADIENT_CLIP_THRESHOLD) {
                    g = (g > 0) ? GRADIENT_CLIP_THRESHOLD : -GRADIENT_CLIP_THRESHOLD;
                }
                // rule of gradient descent
                for (std::size_t j = 0; j < neuronn.weights.size(); ++j) {
                    neuronn.weights[j] -= g * prevlayer_predicts[j];
                }
                neuronn.bias -= g;
            }

            _HIGH_PERF double
            _calc_loss_gradient(const double& predict, const double& target) const noexcept {
                return (predict - target) * dx_swish(predict);
            }

            _HIGH_PERF_VOID void
            _layer_backward(std::vector<Neuron>& mainlayer, const std::vector<double>& prevlayer_predicts, const std::vector<double>& target) noexcept {
                for (std::size_t i = 0; i < mainlayer.size(); ++i) {
                    _update_wb(
                        mainlayer[i],
                        prevlayer_predicts,
                        _calc_loss_gradient(mainlayer[i].predicted, target[i])
                    );
                }
            }
 
            _HIGH_PERF_VOID void
            backward(const Dataset& data) noexcept {
                std::vector<Neuron>& outputLayer = layout.NetworkLayers.outputLayer;
                std::vector<std::vector<Neuron>>& hiddenLayers = layout.NetworkLayers.hiddenLayers;
                
                if (hiddenLayers.empty()) {
                    _layer_backward(
                        outputLayer,
                        data.input,
                        data.target
                    );
                } else {
                    // first step
                    _layer_backward(
                        outputLayer,
                        _get_predicts(hiddenLayers.back()), 
                        data.target
                    );

                    // hidden layers
                    // from x-1 layer
                    for (int i = hiddenLayers.size() - 2; i > 0; --i) {
                        _layer_backward(
                            hiddenLayers[i],
                            _get_predicts(hiddenLayers[i - 1]),
                            data.target
                        );
                    }

                    // last step
                    _layer_backward(
                        hiddenLayers.front(),
                        data.input,
                        data.target
                    );
                }
            }
            
        public:
            explicit DeepNetwork(DeepLayers layers) noexcept : layout(layers) {}

            void
            Train(const std::vector<Dataset>& dataset, const int& epochs) {
                for (int epoch = 0; epoch < epochs; ++epoch) {
                    for (const Dataset& data : dataset) {
                        forward(data.input);
                        backward(data);
                    }
                }
            }

            [[nodiscard]] std::vector<double>
            GetOutput(const std::vector<double>& inputs) noexcept {
                forward(inputs);
                std::vector<double> outs;
                for (std::size_t i = 0; i < layout.NetworkLayers.outputLayer.size(); i++) {
                    outs.push_back(layout.NetworkLayers.outputLayer[i].predicted);
                }
                return outs;
            }
    };
}