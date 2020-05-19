using System;
using System.Linq;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Extensions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;

namespace GingerbreadAI.Model.NeuralNetwork.Test
{
    public class LayerExtensionsShould
    {
        private readonly Layer _input;
        private readonly Layer _hidden1;
        private readonly Layer _hidden2;
        private readonly Layer _output;

        public LayerExtensionsShould()
        {
            _input = new Layer(5, new Layer[0], ActivationFunctionType.Sigmoid, InitialisationFunctionType.Random);
            _hidden1 = new Layer(10, new[] { _input }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.Random);
            _hidden2 = new Layer(15, new[] { _input }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.Random);
            _output = new Layer(20, new[] { _hidden1, _hidden2 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.Random);
        }

        //[Fact]
        public void DeepCopyCorrectly()
        {
            _output.Initialise(new Random());
            _output.CalculateOutputs(new[] { 0.1, 0.2, 0.3, 0.4, 0.5 });

            var copiedOutput = _output.DeepCopy();
            var copiedHidden1 = copiedOutput.PreviousLayers.FirstOrDefault(l => l.Nodes.Length == 10);
            var copiedHidden2 = copiedOutput.PreviousLayers.FirstOrDefault(l => l.Nodes.Length == 15);
            var copiedInput = copiedHidden1.PreviousLayers.Single();

            // input
            for (var i = 0; i < _input.Nodes.Length; i++)
            {
                Assert.Equal(_input.Nodes[i].Output, copiedInput.Nodes[i].Output);
            }
            // hidden 1
            for (var i = 0; i < _hidden1.Nodes.Length; i++)
            {
                Assert.Equal(_hidden1.Nodes[i].Output, copiedHidden1.Nodes[i].Output);
                for (var j = 0; j < _hidden1.Nodes[i].Weights.Count; j++)
                {
                    Assert.Equal(_hidden1.Nodes[i].Weights.Values.ToArray()[j].Value, copiedHidden1.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _hidden1.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.Equal(_hidden1.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedHidden1.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
            }
            // hidden 2
            for (var i = 0; i < _hidden2.Nodes.Length; i++)
            {
                Assert.Equal(_hidden2.Nodes[i].Output, copiedHidden2.Nodes[i].Output);
                for (var j = 0; j < _hidden2.Nodes[i].Weights.Count; j++)
                {
                    Assert.Equal(_hidden2.Nodes[i].Weights.Values.ToArray()[j].Value, copiedHidden2.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _hidden2.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.Equal(_hidden2.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedHidden2.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
            }
            // output
            for (var i = 0; i < _output.Nodes.Length; i++)
            {
                Assert.Equal(_output.Nodes[i].Output, copiedOutput.Nodes[i].Output);
                for (var j = 0; j < _output.Nodes[i].Weights.Count; j++)
                {
                    Assert.Equal(_output.Nodes[i].Weights.Values.ToArray()[j].Value, copiedOutput.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _output.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.Equal(_output.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedOutput.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
            }
        }

        [Fact]
        public void CloneWithSameWeightValueReferencesCorrectly()
        {
            var random = new Random();
            var inputs = new[] { 0.1, 0.2, 0.3, 0.4, 0.5 };

            _output.Initialise(random);
            _output.CalculateOutputs(inputs);

            var copiedOutput = _output.CloneWithDifferentOutputs();
            copiedOutput.Initialise(random);
            copiedOutput.CalculateOutputs(inputs);

            var copiedHidden1 = copiedOutput.PreviousLayers.FirstOrDefault(l => l.Nodes.Length == 10);
            var copiedHidden2 = copiedOutput.PreviousLayers.FirstOrDefault(l => l.Nodes.Length == 15);
            var copiedInput = copiedHidden1.PreviousLayers.Single();

            // input
            Assert.Equal(_input.ActivationFunctionType, copiedInput.ActivationFunctionType);
            Assert.Equal(_input.InitialisationFunctionType, copiedInput.InitialisationFunctionType);
            for (var i = 0; i < _input.Nodes.Length; i++)
            {
                // inputs will be equal (as we supply them!)
                Assert.Equal(_input.Nodes[i].Output, copiedInput.Nodes[i].Output);
            }
            // hidden 1
            Assert.Equal(_hidden1.ActivationFunctionType, copiedHidden1.ActivationFunctionType);
            Assert.Equal(_hidden1.InitialisationFunctionType, copiedHidden1.InitialisationFunctionType);
            for (var i = 0; i < _hidden1.Nodes.Length; i++)
            {
                Assert.NotEqual(_hidden1.Nodes[i].Output, copiedHidden1.Nodes[i].Output);
                for (var j = 0; j < _hidden1.Nodes[i].Weights.Count; j++)
                {
                    Assert.NotSame(_hidden1.Nodes[i].Weights.Keys.ToArray()[j], copiedHidden1.Nodes[i].Weights.Keys.ToArray()[j]);
                    Assert.Equal(_hidden1.Nodes[i].Weights.Values.ToArray()[j].Value, copiedHidden1.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _hidden1.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.NotSame(_hidden1.Nodes[i].BiasWeights.Keys.ToArray()[j], copiedHidden1.Nodes[i].BiasWeights.Keys.ToArray()[j]);
                    Assert.Equal(_hidden1.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedHidden1.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
            }
            // hidden 2
            Assert.Equal(_hidden2.ActivationFunctionType, copiedHidden2.ActivationFunctionType);
            Assert.Equal(_hidden2.InitialisationFunctionType, copiedHidden2.InitialisationFunctionType);
            for (var i = 0; i < _hidden2.Nodes.Length; i++)
            {
                Assert.NotEqual(_hidden2.Nodes[i].Output, copiedHidden2.Nodes[i].Output);
                for (var j = 0; j < _hidden2.Nodes[i].Weights.Count; j++)
                {
                    Assert.NotSame(_hidden2.Nodes[i].Weights.Keys.ToArray()[j], copiedHidden2.Nodes[i].Weights.Keys.ToArray()[j]);
                    Assert.Equal(_hidden2.Nodes[i].Weights.Values.ToArray()[j].Value, copiedHidden2.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _hidden2.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.NotSame(_hidden2.Nodes[i].BiasWeights.Keys.ToArray()[j], copiedHidden2.Nodes[i].BiasWeights.Keys.ToArray()[j]);
                    Assert.Equal(_hidden2.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedHidden2.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
            }
            // output
            Assert.Equal(_output.ActivationFunctionType, copiedOutput.ActivationFunctionType);
            Assert.Equal(_output.InitialisationFunctionType, copiedOutput.InitialisationFunctionType);
            for (var i = 0; i < _output.Nodes.Length; i++)
            {
                Assert.NotEqual(_output.Nodes[i].Output, copiedOutput.Nodes[i].Output);
                for (var j = 0; j < _output.Nodes[i].Weights.Count; j++)
                {
                    Assert.NotSame(_output.Nodes[i].Weights.Keys.ToArray()[j], copiedOutput.Nodes[i].Weights.Keys.ToArray()[j]);
                    Assert.Equal(_output.Nodes[i].Weights.Values.ToArray()[j].Value, copiedOutput.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _output.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.NotSame(_output.Nodes[i].BiasWeights.Keys.ToArray()[j], copiedOutput.Nodes[i].BiasWeights.Keys.ToArray()[j]);
                    Assert.Equal(_output.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedOutput.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
            }
        }
    }
}
