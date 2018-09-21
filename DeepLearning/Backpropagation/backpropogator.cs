namespace Backpropagation
{
    using System.Collections.Generic;
    using System.Linq;
    using AI.Calculations.Calculations;
    using AI.Calculations.Extensions;
    using NeuralNetwork;
    using NeuralNetwork.Data;

    public class Backpropagator
    {
        public double LearningRate { get; set; }

        public LayerCalculator LayerCalculator { get; set; }

        public Backpropagator(Layer outputLayer, double learningRate)
        {
            LayerCalculator = new LayerCalculator
            {
                OutputLayer = outputLayer
            };
            LearningRate = learningRate;
        }

        public void Backpropagate(double[] inputs, double[] targetOutputs)
        {
            var currentLayer = LayerCalculator.OutputLayer;
            var currentOutputs = LayerCalculator.GetResults(inputs);

            var deltas = new Dictionary<Node, double>();

            // initial calculations for output layer
            currentLayer.Nodes.Each((node, i) =>
            {
                var delta = BackpropagationCalculations.GetDeltaOutput(currentOutputs[i], targetOutputs[i]);
                deltas.Add(node, delta);
                foreach (var prevNode in node.Weights.Keys.ToList())
                {
                    node.Weights[prevNode] = node.Weights[prevNode] - (LearningRate * delta * prevNode.Output);
                }

                foreach (var prevLayer in node.BiasWeights.Keys.ToList())
                {
                    node.BiasWeights[prevLayer] = node.BiasWeights[prevLayer] - (LearningRate * delta);
                }
            });

            foreach (var prevLayer in currentLayer.PreviousLayers)
            {
                RecurseBackpropagation(prevLayer, deltas);
            }
        }

        private void RecurseBackpropagation(Layer layer, Dictionary<Node, double> backwardsPassDeltas)
        {
            if (layer.PreviousLayers.Length == 0)
            {
                // input case
                return;
            }

            var deltas = new Dictionary<Node, double>();
            foreach (var node in layer.Nodes)
            {
                var sumDeltaWeights = (double)0;
                foreach (var backPassNode in backwardsPassDeltas.Keys)
                {
                    sumDeltaWeights += backwardsPassDeltas[backPassNode] * backPassNode.Weights[node];
                }
                var delta = sumDeltaWeights * NetworkCalculations.LogisticFunctionDifferential(node.Output);
                deltas.Add(node, delta);

                foreach (var prevNode in node.Weights.Keys.ToList())
                {
                    node.Weights[prevNode] -= (LearningRate * delta * prevNode.Output);
                }

                foreach (var prevLayer in node.BiasWeights.Keys.ToList())
                {
                    node.BiasWeights[prevLayer] -= (LearningRate * delta);
                }
            }

            foreach (var prevLayer in layer.PreviousLayers)
            {
                RecurseBackpropagation(prevLayer, deltas);
            }
        }
    }
}