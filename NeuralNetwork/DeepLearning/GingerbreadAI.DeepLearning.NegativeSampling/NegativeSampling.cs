using System.Collections.Generic;
using System.Linq;
using Model.NeuralNetwork.Models;

namespace DeepLearning.NegativeSampling
{
    public static class NegativeSampling
    {
        public static void NegativeSample(this Layer outputLayer, int inputIndex, int outputIndex, double learningRate, bool isPositiveTarget)
        {
            outputLayer.CalculateIndexedOutput(inputIndex, outputIndex, 1);
            var targetOutput = isPositiveTarget ? 1 : 0;

            var deltas = NegativeSampleOutput(outputLayer, targetOutput, outputIndex, learningRate);

            foreach (var previousLayer in outputLayer.PreviousLayers)
            {
                foreach (var previousPreviousLayer in previousLayer.PreviousLayers)
                {
                    RecurseNegativeSample(previousLayer, previousPreviousLayer, deltas, inputIndex);
                }
            }
        }

        private static Dictionary<Node, double> NegativeSampleOutput(Layer outputLayer, double targetOutput, int outputIndex, double learningRate)
        {
            var outputNode = outputLayer.Nodes[outputIndex];

            var delta = (outputNode.Output - targetOutput)
                        * outputLayer.ActivationFunctionDifferential(outputNode.Output)
                        * learningRate;
            foreach (var weight in outputNode.Weights)
            {
                UpdateNodeWeight(outputNode, weight.Key, weight.Value, delta);
            }
            foreach (var biasWeight in outputNode.BiasWeights)
            {
                UpdateBiasNodeWeight(outputNode, biasWeight.Key, biasWeight.Value, delta);
            }

            return new Dictionary<Node, double> { { outputNode, delta } };
        }

        private static void NegativeSampleInput(Layer layer, Layer inputLayer, Dictionary<Node, double> backwardsPassDeltas, int inputIndex)
        {
            var sumDeltaWeights = (double)0;
            foreach (var backPassDelta in backwardsPassDeltas)
            {
                sumDeltaWeights += backPassDelta.Value;
            }

            var inputNode = inputLayer.Nodes[inputIndex];
            foreach (var node in layer.Nodes)
            {
                var delta = sumDeltaWeights * layer.ActivationFunctionDifferential(node.Output);
                UpdateNodeWeight(node, inputNode, node.Weights[inputNode], delta);
                UpdateBiasNodeWeight(node, inputLayer, node.BiasWeights[inputLayer], delta);
            }
        }

        private static void RecurseNegativeSample(Layer layer, Layer previousLayer, Dictionary<Node, double> backwardsPassDeltas, int inputIndex)
        {
            if (!previousLayer.PreviousLayers.Any())
            {
                NegativeSampleInput(layer, previousLayer, backwardsPassDeltas, inputIndex);
                return;
            }

            var deltas = new Dictionary<Node, double>();
            foreach (var node in layer.Nodes)
            {
                var sumDeltaWeights = (double)0;

                foreach (var backPassNode in backwardsPassDeltas.Keys)
                {
                    sumDeltaWeights += backwardsPassDeltas[backPassNode] * backPassNode.Weights[node].Value;
                }
                var delta = sumDeltaWeights * layer.ActivationFunctionDifferential(node.Output);
                deltas.Add(node, delta);

                foreach (var weight in node.Weights)
                {
                    UpdateNodeWeight(node, weight.Key, weight.Value, delta);
                }

                foreach (var biasWeight in node.BiasWeights)
                {
                    UpdateBiasNodeWeight(node, biasWeight.Key, biasWeight.Value, delta);
                }
            }

            foreach (var prevPrevLayer in previousLayer.PreviousLayers)
            {
                RecurseNegativeSample(previousLayer, prevPrevLayer, deltas, inputIndex);
            }
        }

        private static void UpdateNodeWeight(Node node, Node prevNode, Weight prevNodeWeight, double delta)
        {
            var change = -(delta * prevNode.Output);
            prevNodeWeight.Value += change;
        }

        private static void UpdateBiasNodeWeight(Node node, Layer prevLayer, Weight prevLayerWeight, double delta)
        {
            var change = -delta;
            prevLayerWeight.Value += change;
        }
    }
}
