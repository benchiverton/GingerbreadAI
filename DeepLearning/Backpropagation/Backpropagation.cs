namespace Backpropagation
{
    using System.Collections.Generic;
    using System.Linq;
    using Bens.WonderfulLibrary.Calculations;
    using Bens.WonderfulLibrary.Extensions;
    using NeuralNetwork;
    using NeuralNetwork.Data;

    public class Backpropagation
    {
        public double LearningRate { get; set; }

        public LayerComputor LayerComputor { get; set; }

        public Backpropagation(Layer outputLayer, double learningRate)
        {
            LayerComputor = new LayerComputor
            {
                OutputLayer = outputLayer
            };
            LearningRate = learningRate;
        }

        public void Backpropagate(double[] inputs, double[] targetOutputs)
        {
            var currentLayer = LayerComputor.OutputLayer;
            var curretOutputs = LayerComputor.GetResults(inputs);

            var deltas = new Dictionary<Node, double>();

            // initial calculations for output layer
            currentLayer.Nodes.Each((node, i) =>
            {
                var delta = BackpropagationCalculations.GetDeltaOutput(curretOutputs[i], targetOutputs[i]);
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
                    node.Weights[prevNode] = node.Weights[prevNode] - (LearningRate * delta * prevNode.Output);
                }

                foreach (var prevLayer in node.BiasWeights.Keys.ToList())
                {
                    node.BiasWeights[prevLayer] = node.BiasWeights[prevLayer] - (LearningRate * delta);
                }
            }

            foreach (var prevLayer in layer.PreviousLayers)
            {
                RecurseBackpropagation(prevLayer, deltas);
            }
        }
    }
}