using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using AI.Calculations;
using NeuralNetwork;
using NeuralNetwork.Data;

namespace NegativeSampling
{
    public class NegativeSampler
    {
        private readonly OutputCalculator _outputCalculator;
        private readonly Func<double, double> _learningRateModifier;

        private double _learningRate;

        public NegativeSampler(Layer outputLayer, double learningRate, Func<double, double> learningAction = null)
        {
            _outputCalculator = new OutputCalculator(outputLayer);
            _learningRate = learningRate;
            _learningRateModifier = learningAction;
        }

        public void NegativeSample(int inputIndex, int outputIndex, bool isPositiveTarget)
        {
            var outputLayer = _outputCalculator.OutputLayer;
            var currentOutput = _outputCalculator.GetResult(inputIndex, outputIndex);
            var targetOutput = isPositiveTarget ? 1 : 0;

            var deltas = NegativeSampleOutput(outputLayer, currentOutput, targetOutput, outputIndex);

            for (var i = 0; i < outputLayer.PreviousLayers.Length; i++)
            {
                for (var j = 0; j < outputLayer.PreviousLayers[i].PreviousLayers.Length; j++)
                {
                    RecurseNegativeSample(outputLayer.PreviousLayers[i], outputLayer.PreviousLayers[i].PreviousLayers[j], deltas, inputIndex);
                }
            }

            if (_learningRateModifier != null)
            {
                _learningRate = _learningRateModifier(_learningRate);
            }
        }

        private Dictionary<Node, double> NegativeSampleOutput(Layer outputLayer, double currentOutput, double targetOutput, int outputIndex)
        {
            var outputNode = outputLayer.Nodes[outputIndex];

            var delta = BackpropagationCalculations.GetDeltaOutput(currentOutput, targetOutput);
            foreach (var previousNode in outputNode.Weights.Keys.ToList())
            {
                UpdateNodeWeight(outputNode, previousNode, delta);
            }
            foreach (var previousBiasLayer in outputNode.BiasWeights.Keys.ToList())
            {
                UpdateBiasNodeWeight(outputNode, previousBiasLayer, delta);
            }

            return new Dictionary<Node, double> { { outputNode, delta } };
        }

        private void NegativeSampleInput(Layer layer, Layer inputLayer, Dictionary<Node, double> backwardsPassDeltas, int inputIndex)
        {
            var inputNode = inputLayer.Nodes[inputIndex];
            foreach (var node in layer.Nodes)
            {
                var sumDeltaWeights = (double)0;
                foreach (var backPassNode in backwardsPassDeltas.Keys)
                {
                    sumDeltaWeights += backPassNode.Output * backPassNode.Weights[node].Value;
                }
                var delta = sumDeltaWeights * NetworkCalculations.LogisticFunctionDifferential(node.Output);
                UpdateNodeWeight(node, inputNode, delta);
                UpdateBiasNodeWeight(node, inputLayer, delta);
            }
        }

        private void RecurseNegativeSample(Layer layer, Layer previousLayer, Dictionary<Node, double> backwardsPassDeltas, int inputIndex)
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
                    sumDeltaWeights += backPassNode.Output * backPassNode.Weights[node].Value;
                }
                var delta = sumDeltaWeights * NetworkCalculations.LogisticFunctionDifferential(node.Output);
                deltas.Add(node, delta);

                foreach (var prevNode in node.Weights.Keys.ToList())
                {
                    UpdateNodeWeight(node, prevNode, delta);
                }

                foreach (var prevLayer in node.BiasWeights.Keys.ToList())
                {
                    UpdateBiasNodeWeight(node, prevLayer, delta);
                }
            }

            foreach (var prevPrevLayer in previousLayer.PreviousLayers)
            {
                RecurseNegativeSample(previousLayer, prevPrevLayer, deltas, inputIndex);
            }
        }

        private void UpdateNodeWeight(Node node, Node prevNode, double delta)
        {
            var change = -(_learningRate * delta * prevNode.Output);
            node.Weights[prevNode].Value += change;
        }

        private void UpdateBiasNodeWeight(Node node, Layer prevLayer, double delta)
        {
            var change = -(_learningRate * delta);
            node.BiasWeights[prevLayer].Value += change;
        }
    }
}
