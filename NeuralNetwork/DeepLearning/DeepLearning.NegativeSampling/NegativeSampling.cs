using System.Collections.Generic;
using System.Linq;
using Library.Computations;
using Model.NeuralNetwork;
using Model.NeuralNetwork.Models;

namespace DeepLearning.NegativeSampling
{
    public static class NegativeSampling
    {
        public static void NegativeSample(this Layer outputLayer, int inputIndex, int outputIndex, double learningRate, bool isPositiveTarget)
        {
            var currentOutput = outputLayer.GetResult(inputIndex, outputIndex);
            var targetOutput = isPositiveTarget ? 1 : 0;

            var deltas = NegativeSampleOutput(outputLayer, currentOutput, targetOutput, outputIndex, learningRate);

            foreach (var previousLayer in outputLayer.PreviousLayers)
            {
                foreach (var previousPreviousLayer in previousLayer.PreviousLayers)
                {
                    RecurseNegativeSample(previousLayer, previousPreviousLayer, deltas, inputIndex);
                }
            }
        }

        private static Dictionary<Node, double> NegativeSampleOutput(Layer outputLayer, double currentOutput, double targetOutput, int outputIndex, double learningRate)
        {
            var outputNode = outputLayer.Nodes[outputIndex];

            var delta = LogisticFunction.ComputeDeltaOutput(currentOutput, targetOutput) * learningRate;
            foreach (var previousNode in outputNode.Weights.Keys)
            {
                UpdateNodeWeight(outputNode, previousNode, delta);
            }
            foreach (var previousBiasLayer in outputNode.BiasWeights.Keys)
            {
                UpdateBiasNodeWeight(outputNode, previousBiasLayer, delta);
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
                var delta = sumDeltaWeights * LogisticFunction.ComputeDifferentialGivenOutput(node.Output);
                UpdateNodeWeight(node, inputNode, delta);
                UpdateBiasNodeWeight(node, inputLayer, delta);
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
                var delta = sumDeltaWeights * LogisticFunction.ComputeDifferentialGivenOutput(node.Output);
                deltas.Add(node, delta);

                foreach (var prevNode in node.Weights.Keys)
                {
                    UpdateNodeWeight(node, prevNode, delta);
                }

                foreach (var prevLayer in node.BiasWeights.Keys)
                {
                    UpdateBiasNodeWeight(node, prevLayer, delta);
                }
            }

            foreach (var prevPrevLayer in previousLayer.PreviousLayers)
            {
                RecurseNegativeSample(previousLayer, prevPrevLayer, deltas, inputIndex);
            }
        }

        private static void UpdateNodeWeight(Node node, Node prevNode, double delta)
        {
            var change = -(delta * prevNode.Output);
            node.Weights[prevNode].Value += change;
        }

        private static void UpdateBiasNodeWeight(Node node, Layer prevLayer, double delta)
        {
            var change = -delta;
            node.BiasWeights[prevLayer].Value += change;
        }
    }
}
