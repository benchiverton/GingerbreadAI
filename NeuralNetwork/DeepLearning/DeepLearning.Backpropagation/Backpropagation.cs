using System.Collections.Generic;
using System.Linq;
using Model.NeuralNetwork;
using Model.NeuralNetwork.Models;

namespace DeepLearning.Backpropagation
{
    public static class Backpropagation
    {
        public static void Backpropagate(this Layer outputLayer, double[] inputs, double[] targetOutputs, double learningRate, Layer momentum = null, double momentumMagnitude = 0d)
        {
            var currentOutputs = outputLayer.GetResults(inputs);

            DoBackpropagation(outputLayer, currentOutputs, targetOutputs, learningRate, momentum, momentumMagnitude);
        }

        public static void Backpropagate(this Layer outputLayer, Dictionary<Layer, double[]> inputs, double[] targetOutputs, double learningRate, Layer momentum = null, double momentumMagnitude = 0d)
        {
            var currentOutputs = outputLayer.GetResults(inputs);

            DoBackpropagation(outputLayer, currentOutputs, targetOutputs, learningRate, momentum, momentumMagnitude);
        }

        private static void DoBackpropagation(Layer outputLayer, double[] currentOutputs, double[] targetOutputs, double learningRate, Layer momentum, double momentumMagnitude)
        {
            var backwardsPassDeltas = UpdateOutputLayer(outputLayer, currentOutputs, targetOutputs, learningRate, momentum, momentumMagnitude);

            for (var i = 0; i < outputLayer.PreviousLayers.Length; i++)
            {
                RecurseBackpropagation(outputLayer.PreviousLayers[i], backwardsPassDeltas, momentum?.PreviousLayers[i], momentumMagnitude);
            }
        }

        private static void RecurseBackpropagation(Layer layer, Dictionary<Node, double> backwardsPassDeltas, Layer momentum, double momentumMagnitude)
        {
            if (!layer.PreviousLayers.Any())
            {
                // input case
                return;
            }

            var deltas = new Dictionary<Node, double>();
            for (var i = 0; i < layer.Nodes.Length; i++)
            {
                var node = layer.Nodes[i];
                var momentumNode = momentum?.Nodes[i];
                var sumDeltaWeights = backwardsPassDeltas.Sum(
                    backwardsPassDelta => backwardsPassDelta.Key.Weights.TryGetValue(node, out var backPassWeight)
                        ? backwardsPassDelta.Value * backPassWeight.Value
                        : 0d);
                var delta = sumDeltaWeights * layer.ActivationFunctionDifferential(node.Output);
                deltas.Add(node, delta);

                foreach (var weight in node.Weights)
                {
                    UpdateNodeWeight(weight.Key, momentumNode, weight.Value, delta, momentumMagnitude);
                }

                foreach (var biasWeight in node.BiasWeights)
                {
                    UpdateBiasNodeWeight(biasWeight.Key, momentumNode, biasWeight.Value, delta, momentumMagnitude);
                }
            }

            for (var i = 0; i < layer.PreviousLayers.Length; i++)
            {
                RecurseBackpropagation(layer.PreviousLayers[i], deltas, momentum?.PreviousLayers[i], momentumMagnitude);
            }
        }

        private static Dictionary<Node, double> UpdateOutputLayer(Layer outputLayer, double[] currentOutputs, double[] targetOutputs, double learningRate, Layer momentum, double momentumMagnitude)
        {
            var deltas = new Dictionary<Node, double>();

            for (var i = 0; i < outputLayer.Nodes.Length; i++)
            {
                var node = outputLayer.Nodes[i];
                var momentumNode = momentum?.Nodes[i];
                var delta = (currentOutputs[i] - targetOutputs[i]) 
                            * outputLayer.ActivationFunctionDifferential(currentOutputs[i]) 
                            * learningRate;
                deltas.Add(node, delta);
                foreach (var weight in node.Weights)
                {
                    UpdateNodeWeight(weight.Key, momentumNode, weight.Value, delta, momentumMagnitude);
                }

                foreach (var biasWeight in node.BiasWeights)
                {
                    UpdateBiasNodeWeight(biasWeight.Key, momentumNode, biasWeight.Value, delta, momentumMagnitude);
                }
            }

            return deltas;
        }

        private static void UpdateNodeWeight(Node prevNode, Node momentumNode, Weight prevNodeWeight, double delta, double momentumMagnitude)
        {
            var change = -(delta * prevNode.Output);
            prevNodeWeight.Value += change;

            if (momentumNode == null) return;

            var momentumWeight = momentumNode.Weights[prevNode];
            var changeFromMomentum = momentumMagnitude * momentumWeight.Value;
            prevNodeWeight.Value += changeFromMomentum;
            momentumWeight.Value = change + changeFromMomentum;
        }

        private static void UpdateBiasNodeWeight(Layer prevLayer, Node momentumNode, Weight prevLayerWeight, double delta, double momentumMagnitude)
        {
            var change = -delta;
            prevLayerWeight.Value += change;

            if (momentumNode == null) return;

            var momentumWeight = momentumNode.BiasWeights[prevLayer];
            var changeFromMomentum = momentumMagnitude * momentumWeight.Value;
            prevLayerWeight.Value += changeFromMomentum;
            momentumWeight.Value = change + changeFromMomentum;
        }
    }
}
