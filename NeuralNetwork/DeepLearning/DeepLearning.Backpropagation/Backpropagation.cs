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
                var sumDeltaWeights = backwardsPassDeltas.Sum(
                    backwardsPassDelta => backwardsPassDelta.Key.Weights.TryGetValue(node, out var backPassWeight)
                        ? backwardsPassDelta.Value * backPassWeight.Value
                        : 0d);
                var delta = sumDeltaWeights * LogisticFunction.ComputeDifferentialGivenOutput(node.Output);
                deltas.Add(node, delta);

                foreach (var weight in node.Weights)
                {
                    UpdateNodeWeight(node, weight.Key, weight.Value, delta, momentum, i);
                }

                foreach (var biasWeight in node.BiasWeights)
                {
                    UpdateBiasNodeWeight(node, biasWeight.Key, biasWeight.Value, delta, momentum, i);
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
                foreach (var weight in node.Weights)
                {
                    UpdateNodeWeight(node, weight.Key, weight.Value, delta, momentum, i);
                }

                foreach (var biasWeight in node.BiasWeights)
                {
                    UpdateBiasNodeWeight(node, biasWeight.Key, biasWeight.Value, delta, momentum, i);
                }
            }

            return deltas;
        }

        private static void UpdateNodeWeight(Node node, Node prevNode, Weight weight, double delta, Momentum momentum, int nodeIndex)
        {
            var change = -(delta * prevNode.Output);
            weight.Value += change;

            momentum?.ApplyMomentum(node, prevNode, change, nodeIndex);
        }

        private static void UpdateBiasNodeWeight(Node node, Layer prevLayer, Weight weight, double delta, Momentum momentum, int nodeIndex)
        {
            var change = -delta;
            weight.Value += change;

            momentum?.ApplyBiasMomentum(node, prevLayer, change, nodeIndex);
        }
    }
}
