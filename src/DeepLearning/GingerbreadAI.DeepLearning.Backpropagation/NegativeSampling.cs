using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.DeepLearning.Backpropagation
{
    public static class NegativeSampling
    {
        public static void NegativeSample(this Layer outputLayer, int inputIndex, int outputIndex, bool isPositiveTarget, ErrorFunctionType errorFunctionType, double learningRate, double momentumMagnitude = 0d)
        {
            outputLayer.CalculateIndexedOutput(inputIndex, outputIndex, 1);
            var targetOutput = isPositiveTarget ? 1 : 0;

            var deltas = NegativeSampleOutput(outputLayer, targetOutput, outputIndex, errorFunctionType, learningRate, momentumMagnitude);

            foreach (var previousLayer in outputLayer.PreviousLayers)
            {
                foreach (var previousPreviousLayer in previousLayer.PreviousLayers)
                {
                    RecurseNegativeSample(previousLayer, previousPreviousLayer, inputIndex, deltas, momentumMagnitude);
                }
            }
        }

        private static Dictionary<Node, double> NegativeSampleOutput(Layer outputLayer, double targetOutput, int outputIndex, ErrorFunctionType errorFunctionType, double learningRate, double momentumMagnitude)
        {
            var errorFunctionDifferential = ErrorFunctionResolver.ResolveErrorFunctionDifferential(errorFunctionType);
            var outputNode = outputLayer.Nodes[outputIndex];

            var delta = errorFunctionDifferential.Invoke(targetOutput, outputNode.Output)
                        * outputLayer.ActivationFunctionDifferential(outputNode.Output)
                        * learningRate;
            foreach (var (prevNode, weightForPrevNode) in outputNode.Weights)
            {
                Backpropagation.UpdateNodeWeight(prevNode, weightForPrevNode, delta, momentumMagnitude);
            }
            foreach (var (_, weightForPrevLayer) in outputNode.BiasWeights)
            {
                Backpropagation.UpdateBiasNodeWeight(weightForPrevLayer, delta, momentumMagnitude);
            }

            return new Dictionary<Node, double> { { outputNode, delta } };
        }

        private static void RecurseNegativeSample(Layer layer, Layer previousLayer, int inputIndex, Dictionary<Node, double> backwardsPassDeltas, double momentumMagnitude)
        {
            if (!previousLayer.PreviousLayers.Any())
            {
                // case where previous layer is input
                NegativeSampleFirstHiddenLayer(layer, previousLayer, inputIndex, backwardsPassDeltas, momentumMagnitude);
                return;
            }

            var deltas = new Dictionary<Node, double>();
            foreach (var node in layer.Nodes)
            {
                var delta = Backpropagation.CalculateDelta(layer, backwardsPassDeltas, node);
                deltas.Add(node, delta);

                foreach (var (prevNode, weightForPrevNode) in node.Weights)
                {
                    Backpropagation.UpdateNodeWeight(prevNode, weightForPrevNode, delta, momentumMagnitude);
                }

                foreach (var (_, weightForPrevLayer) in node.BiasWeights)
                {
                    Backpropagation.UpdateBiasNodeWeight(weightForPrevLayer, delta, momentumMagnitude);
                }
            }

            foreach (var prevPrevLayer in previousLayer.PreviousLayers)
            {
                RecurseNegativeSample(previousLayer, prevPrevLayer, inputIndex, deltas, momentumMagnitude);
            }
        }

        private static void NegativeSampleFirstHiddenLayer(Layer layer, Layer inputLayer, int inputIndex, Dictionary<Node, double> backwardsPassDeltas, double momentumMagnitude)
        {
            var inputNode = inputLayer.Nodes[inputIndex];
            foreach (var node in layer.Nodes)
            {
                var delta = Backpropagation.CalculateDelta(layer, backwardsPassDeltas, node);
                Backpropagation.UpdateNodeWeight(inputNode, node.Weights[inputNode], delta, momentumMagnitude);
                if (node.BiasWeights.TryGetValue(inputLayer, out var biasWeight))
                {
                    Backpropagation.UpdateBiasNodeWeight(biasWeight, delta, momentumMagnitude);
                }
            }
        }
    }
}
