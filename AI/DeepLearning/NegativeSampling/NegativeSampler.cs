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
            var currentLayer = _outputCalculator.OutputLayer;
            var currentOutput = _outputCalculator.GetResult(inputIndex, outputIndex);
            var targetOutput = isPositiveTarget ? 1 : 0;

            Console.WriteLine($"{currentOutput}\t{targetOutput}");

            var delta = BackpropagationCalculations.GetDeltaOutput(currentOutput, targetOutput);
            foreach (var previousNodeWeight in _outputCalculator.OutputLayer.Nodes[outputIndex].Weights)
            {
                UpdateNodeWeight(_outputCalculator.OutputLayer.Nodes[outputIndex], previousNodeWeight.Key, delta);
            }
            foreach (var previousBiasWeight in _outputCalculator.OutputLayer.Nodes[outputIndex].BiasWeights)
            {
                UpdateBiasNodeWeight(_outputCalculator.OutputLayer.Nodes[outputIndex], previousBiasWeight.Key, delta);
            }

            var deltas = new Dictionary<Node, double> { { _outputCalculator.OutputLayer.Nodes[outputIndex], delta } };

            foreach (var previousLayer in _outputCalculator.OutputLayer.PreviousLayers)
            {
                foreach (var previousPreviousLayer in previousLayer.PreviousLayers)
                {
                    RecurseNegativeSample(previousLayer, previousPreviousLayer, deltas, inputIndex);
                }
            }

            if (_learningRateModifier != null)
            {
                _learningRate = _learningRateModifier(_learningRate);
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
                foreach (var backPassNode in previousLayer.Nodes)
                {
                    sumDeltaWeights += backwardsPassDeltas[backPassNode] * backPassNode.Weights[node].Value;
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

        private void NegativeSampleInput(Layer layer, Layer previousLayer, Dictionary<Node, double> backwardsPassDeltas, int inputIndex)
        {
            foreach (var node in layer.Nodes)
            {
                var sumDeltaWeights = (double)0;
                foreach (var backPassNode in backwardsPassDeltas.Keys)
                {
                    sumDeltaWeights += backwardsPassDeltas[backPassNode] * backPassNode.Weights[node].Value;
                }
                var delta = sumDeltaWeights * NetworkCalculations.LogisticFunctionDifferential(node.Output);
                UpdateNodeWeight(node, previousLayer.Nodes[inputIndex], delta);
                UpdateBiasNodeWeight(node, previousLayer, delta);
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
