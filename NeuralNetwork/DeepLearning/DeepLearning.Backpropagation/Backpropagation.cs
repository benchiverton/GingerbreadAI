using System.Collections.Generic;
using System.Linq;
using Library.Computations;
using Model.NeuralNetwork;
using Model.NeuralNetwork.Models;

namespace DeepLearning.Backpropagation
{
    public static class Backpropagation
    {
        public static void Backpropagate(this Layer outputLayer, double[] inputs, double[] targetOutputs, double learningRate, Momentum momentum = null)
        {
            var currentOutputs = outputLayer.GetResults(inputs);

            DoBackpropagation(outputLayer, currentOutputs, targetOutputs, learningRate, momentum);
        }

        public static void Backpropagate(this Layer outputLayer, Dictionary<Layer, double[]> inputs, double[] targetOutputs, double learningRate, Momentum momentum = null)
        {
            var currentOutputs = outputLayer.GetResults(inputs);

            DoBackpropagation(outputLayer, currentOutputs, targetOutputs, learningRate, momentum);
        }

        private static void DoBackpropagation(Layer outputLayer, double[] currentOutputs, double[] targetOutputs, double learningRate, Momentum momentum)
        {
            var backwardsPassDeltas = UpdateOutputLayer(outputLayer, currentOutputs, targetOutputs, learningRate, momentum);

            for (var i = 0; i < outputLayer.PreviousLayers.Length; i++)
            {
                RecurseBackpropagation(outputLayer.PreviousLayers[i], backwardsPassDeltas, momentum?.StepBackwards(i));
            }
        }

        private static void RecurseBackpropagation(Layer layer, Dictionary<Node, double> backwardsPassDeltas, Momentum momentum)
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
                var sumDeltaWeights = backwardsPassDeltas.Keys.Sum(
                    backPassNode => backPassNode.Weights.TryGetValue(node, out var backPassWeight)
                                        ? backwardsPassDeltas[backPassNode] * backPassWeight.Value
                                        : 0
                );
                var delta = sumDeltaWeights * LogisticFunction.ComputeDifferentialGivenOutput(node.Output);
                deltas.Add(node, delta);

                foreach (var prevNode in node.Weights.Keys)
                {
                    UpdateNodeWeight(node, prevNode, delta, momentum, i);
                }

                foreach (var prevLayer in node.BiasWeights.Keys)
                {
                    UpdateBiasNodeWeight(node, prevLayer, delta, momentum, i);
                }
            }

            for (var i = 0; i < layer.PreviousLayers.Length; i++)
            {
                RecurseBackpropagation(layer.PreviousLayers[i], deltas, momentum?.StepBackwards(i));
            }
        }

        private static Dictionary<Node, double> UpdateOutputLayer(Layer outputLayer, double[] currentOutputs, double[] targetOutputs, double learningRate, Momentum momentum)
        {
            var deltas = new Dictionary<Node, double>();

            for (var i = 0; i < outputLayer.Nodes.Length; i++)
            {
                var node = outputLayer.Nodes[i];
                var delta = LogisticFunction.ComputeDeltaOutput(currentOutputs[i], targetOutputs[i]) * learningRate;
                deltas.Add(node, delta);
                foreach (var prevNode in node.Weights.Keys)
                {
                    UpdateNodeWeight(node, prevNode, delta, momentum, i);
                }

                foreach (var prevLayer in node.BiasWeights.Keys)
                {
                    UpdateBiasNodeWeight(node, prevLayer, delta, momentum, i);
                }
            }

            return deltas;
        }

        private static void UpdateNodeWeight(Node node, Node prevNode, double delta, Momentum momentum, int nodeIndex)
        {
            var change = -(delta * prevNode.Output);
            node.Weights[prevNode].Value += change;

            momentum?.ApplyMomentum(node, prevNode, change, nodeIndex);
        }

        private static void UpdateBiasNodeWeight(Node node, Layer prevLayer, double delta, Momentum momentum, int nodeIndex)
        {
            var change = -delta;
            node.BiasWeights[prevLayer].Value += change;

            momentum?.ApplyBiasMomentum(node, prevLayer, change, nodeIndex);
        }
    }
}
