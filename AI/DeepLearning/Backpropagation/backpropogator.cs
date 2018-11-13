namespace Backpropagation
{
    using System.Collections.Generic;
    using System.Linq;
    using AI.Calculations.Calculations;
    using NeuralNetwork;
    using NeuralNetwork.Data;
    using NeuralNetwork.Library.Extensions;

    public class Backpropagator
    {
        private readonly Layer _momentumDeltaHolder;

        public LayerCalculator LayerCalculator { get; set; }

        public double LearningRate { get; set; }

        public double Momentum { get; set; }

        public Backpropagator(Layer outputLayer, double learningRate, double momentum = 0)
        {
            LayerCalculator = new LayerCalculator
            {
                OutputLayer = outputLayer
            };
            LearningRate = learningRate;

            Momentum = momentum;
            _momentumDeltaHolder = outputLayer.GetCopyWithReferences();
        }

        public void Backpropagate(double[] inputs, double[] targetOutputs)
        {
            var currentLayer = LayerCalculator.OutputLayer;
            var currentOutputs = LayerCalculator.GetResults(inputs);

            var deltas = new Dictionary<Node, double>();

            // initial calculations for output layer
            for (var i = 0; i < currentLayer.Nodes.Length; i++)
            {
                var node = currentLayer.Nodes[i];
                var delta = BackpropagationCalculations.GetDeltaOutput(currentOutputs[i], targetOutputs[i]);
                deltas.Add(node, delta);
                foreach (var prevNode in node.Weights.Keys.ToList())
                {
                    UpdateNodeWeight(node, prevNode, delta, _momentumDeltaHolder.Nodes[i]);
                }

                foreach (var prevLayer in node.BiasWeights.Keys.ToList())
                {
                    UpdateBiasNodeWeight(node, prevLayer, delta, _momentumDeltaHolder.Nodes[i]);
                }
            }

            for (var i = 0; i < currentLayer.PreviousLayers.Length; i++)
            {
                RecurseBackpropagation(currentLayer.PreviousLayers[i], deltas, _momentumDeltaHolder.PreviousLayers[i]);
            }
        }

        private void RecurseBackpropagation(Layer layer, Dictionary<Node, double> backwardsPassDeltas, Layer momentumLayer)
        {
            if (layer.PreviousLayers.Length == 0)
            {
                // input case
                return;
            }

            var deltas = new Dictionary<Node, double>();
            for (var i = 0; i < layer.Nodes.Length; i++)
            {
                var node = layer.Nodes[i];
                var sumDeltaWeights = (double)0;
                foreach (var backPassNode in backwardsPassDeltas.Keys)
                {
                    sumDeltaWeights += backwardsPassDeltas[backPassNode] * backPassNode.Weights[node];
                }
                var delta = sumDeltaWeights * NetworkCalculations.LogisticFunctionDifferential(node.Output);
                deltas.Add(node, delta);

                foreach (var prevNode in node.Weights.Keys.ToList())
                {
                    UpdateNodeWeight(node, prevNode, delta, momentumLayer.Nodes[i]);
                }

                foreach (var prevLayer in node.BiasWeights.Keys.ToList())
                {
                    UpdateBiasNodeWeight(node, prevLayer, delta, momentumLayer.Nodes[i]);
                }
            }

            for (var i = 0; i < layer.PreviousLayers.Length; i++)
            {
                RecurseBackpropagation(layer.PreviousLayers[i], deltas, momentumLayer.PreviousLayers[i]);
            }
        }

        private void UpdateNodeWeight(Node node, Node prevNode, double delta, Node momentumNode)
        {
            var change = -(LearningRate * delta * prevNode.Output);
            node.Weights[prevNode] += change;

            // apply momentum
            node.Weights[prevNode] += Momentum * momentumNode.Weights[prevNode];
            momentumNode.Weights[prevNode] = change + Momentum * momentumNode.Weights[prevNode];
        }

        private void UpdateBiasNodeWeight(Node node, Layer prevLayer, double delta, Node momentumNode)
        {
            var change = -(LearningRate * delta);
            node.BiasWeights[prevLayer] += change;

            // apply momentum
            node.BiasWeights[prevLayer] += Momentum * momentumNode.BiasWeights[prevLayer];
            momentumNode.BiasWeights[prevLayer] = change + Momentum * momentumNode.BiasWeights[prevLayer];
        }
    }
}