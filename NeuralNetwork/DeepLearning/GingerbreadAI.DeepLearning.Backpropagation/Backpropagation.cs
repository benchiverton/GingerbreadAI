using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.DeepLearning.Backpropagation.Interfaces;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.DeepLearning.Backpropagation
{
    public static class Backpropagation
    {
        public static void Backpropagate(this Layer outputLayer, double[] inputs, double[] targetOutputs, double learningRate, double momentumMagnitude = 0d)
        {
            outputLayer.CalculateOutputs(inputs);

            DoBackpropagation(outputLayer, targetOutputs, learningRate, momentumMagnitude);
        }

        public static void Backpropagate(this Layer outputLayer, Dictionary<Layer, double[]> inputs, double[] targetOutputs, double learningRate, double momentumMagnitude = 0d)
        {
            outputLayer.CalculateOutputs(inputs);

            DoBackpropagation(outputLayer, targetOutputs, learningRate, momentumMagnitude);
        }

        private static void DoBackpropagation(Layer outputLayer, double[] targetOutputs, double learningRate, double momentumMagnitude)
        {
            var backwardsPassDeltas = UpdateOutputLayer(outputLayer, targetOutputs, learningRate, momentumMagnitude);

            foreach (var t in outputLayer.PreviousLayers)
            {
                RecurseBackpropagation(t, backwardsPassDeltas, momentumMagnitude);
            }
        }

        private static void RecurseBackpropagation(Layer layer, Dictionary<Node, double> backwardsPassDeltas, double momentumMagnitude)
        {
            if (!layer.PreviousLayers.Any())
            {
                // input case
                return;
            }

            var deltas = new Dictionary<Node, double>();
            foreach (var node in layer.Nodes)
            {
                // TODO: optimise this
                var sumDeltaWeights = 0d;
                foreach (var (passedNode, passedDelta) in backwardsPassDeltas)
                {
                    if (passedNode.Weights.TryGetValue(node, out var weightBetweenNodeAndPassedNode))
                    {
                        sumDeltaWeights += passedDelta * weightBetweenNodeAndPassedNode.Value;
                    }
                }
                var delta = sumDeltaWeights * layer.ActivationFunctionDifferential(node.Output);
                deltas.Add(node, delta);

                foreach (var (prevNode, weightForPrevNode) in node.Weights)
                {
                    UpdateNodeWeight(prevNode, weightForPrevNode, delta, momentumMagnitude);
                }

                foreach (var (_, weightForPrevLayer) in node.BiasWeights)
                {
                    UpdateBiasNodeWeight(weightForPrevLayer, delta, momentumMagnitude);
                }
            }

            foreach (var t in layer.PreviousLayers)
            {
                RecurseBackpropagation(t, deltas, momentumMagnitude);
            }
        }

        private static Dictionary<Node, double> UpdateOutputLayer(Layer outputLayer, double[] targetOutputs, double learningRate, double momentumMagnitude)
        {
            var deltas = new Dictionary<Node, double>();

            for (var i = 0; i < outputLayer.Nodes.Length; i++)
            {
                var node = outputLayer.Nodes[i];
                var delta = (node.Output - targetOutputs[i])
                            * outputLayer.ActivationFunctionDifferential(node.Output)
                            * learningRate;
                deltas.Add(node, delta);
                foreach (var (prevNode, weightForPrevNode) in node.Weights)
                {
                    UpdateNodeWeight(prevNode, weightForPrevNode, delta, momentumMagnitude);
                }

                foreach (var (_, weightForPrevLayer) in node.BiasWeights)
                {
                    UpdateBiasNodeWeight(weightForPrevLayer, delta, momentumMagnitude);
                }
            }

            return deltas;
        }

        private static void UpdateNodeWeight(Node prevNode, Weight weightForPrevNode, double delta, double momentumMagnitude)
        {
            var change = -(delta * prevNode.Output);
            weightForPrevNode.Value += change;

            if (weightForPrevNode is IWeightWithMomentum weightForPrevNodeAsWeightWithMomentum)
            {
                var changeFromMomentum = momentumMagnitude * weightForPrevNodeAsWeightWithMomentum.Momentum;
                weightForPrevNode.Value += changeFromMomentum;
                weightForPrevNodeAsWeightWithMomentum.Momentum = change + changeFromMomentum;
            }
        }

        private static void UpdateBiasNodeWeight(Weight weightForPrevLayer, double delta, double momentumMagnitude)
        {
            var change = -delta;
            weightForPrevLayer.Value += change;

            if (weightForPrevLayer is IWeightWithMomentum weightForPrevLayerAsWeightWithMomentum)
            {
                var changeFromMomentum = momentumMagnitude * weightForPrevLayerAsWeightWithMomentum.Momentum;
                weightForPrevLayer.Value += changeFromMomentum;
                weightForPrevLayerAsWeightWithMomentum.Momentum = change + changeFromMomentum;
            }
        }
    }
}
