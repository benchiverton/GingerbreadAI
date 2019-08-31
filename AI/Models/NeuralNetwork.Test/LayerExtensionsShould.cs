namespace NeuralNetwork.Test
{
    using System;
    using System.Linq;
    using NeuralNetwork.Models;
    using Xunit;
    using Xunit.Abstractions;

    public class LayerExtensionsShould
    {
        private readonly ITestOutputHelper _testOutputHelper;

        private readonly Layer _input;
        private readonly Layer _hidden1;
        private readonly Layer _hidden2;
        private readonly Layer _output;

        public LayerExtensionsShould(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;

            _input = new Layer("input", 5, new Layer[0]);
            _hidden1 = new Layer("hidden1", 10, new[] { _input });
            _hidden2 = new Layer("hidden2", 15, new[] { _input });
            _output = new Layer("output", 20, new[] { _hidden1, _hidden2 });

            _output.PopulateAllOutputs(new[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
        }

        [Fact]
        public void DeepCopyCorrectly()
        {
            LayerInitialiser.Initialise(new Random(), _output);
            _testOutputHelper.WriteLine(_output.ToString(true));

            var copiedOutput = _output.DeepCopy();
            var copiedHidden1 = copiedOutput.PreviousLayers.FirstOrDefault(l => l.Name == "hidden1");
            var copiedHidden2 = copiedOutput.PreviousLayers.FirstOrDefault(l => l.Name == "hidden2");
            var copiedInput = copiedHidden1.PreviousLayers.Single();

            // input
            for (var i = 0; i < _input.Nodes.Length; i++)
            {
                for (var j = 0; j < _input.Nodes[i].Weights.Count; j++)
                {
                    Assert.Equal(_input.Nodes[i].Weights.Values.ToArray()[j].Value, copiedInput.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                Assert.Equal(_input.Nodes[i].Output, copiedInput.Nodes[i].Output);
            }
            // hidden 1
            for (var i = 0; i < _hidden1.Nodes.Length; i++)
            {
                for (var j = 0; j < _hidden1.Nodes[i].Weights.Count; j++)
                {
                    Assert.Equal(_hidden1.Nodes[i].Weights.Values.ToArray()[j].Value, copiedHidden1.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _hidden1.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.Equal(_hidden1.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedHidden1.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
                Assert.Equal(_hidden1.Nodes[i].Output, copiedHidden1.Nodes[i].Output);
            }
            // hidden 2
            for (var i = 0; i < _hidden2.Nodes.Length; i++)
            {
                for (var j = 0; j < _hidden2.Nodes[i].Weights.Count; j++)
                {
                    Assert.Equal(_hidden2.Nodes[i].Weights.Values.ToArray()[j].Value, copiedHidden2.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _hidden2.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.Equal(_hidden2.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedHidden2.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
                Assert.Equal(_hidden2.Nodes[i].Output, copiedHidden2.Nodes[i].Output);
            }
            // output
            for (var i = 0; i < _output.Nodes.Length; i++)
            {
                for (var j = 0; j < _output.Nodes[i].Weights.Count; j++)
                {
                    Assert.Equal(_output.Nodes[i].Weights.Values.ToArray()[j].Value, copiedOutput.Nodes[i].Weights.Values.ToArray()[j].Value);
                }
                for (var j = 0; j < _output.Nodes[i].BiasWeights.Count; j++)
                {
                    Assert.Equal(_output.Nodes[i].BiasWeights.Values.ToArray()[j].Value, copiedOutput.Nodes[i].BiasWeights.Values.ToArray()[j].Value);
                }
                Assert.Equal(_output.Nodes[i].Output, copiedOutput.Nodes[i].Output);
            }
        }

        [Fact]
        public void CloneWithNodeReferencesCorrectly()
        {
            // TODO
        }

        [Fact]
        public void CloneWithNodeAndWeightReferencesCorrectly()
        {
            // TODO
        }

        [Fact(Skip = "Debug only")]
        public void SaveCorrectly()
        {
            // TODO
        }
    }
}
