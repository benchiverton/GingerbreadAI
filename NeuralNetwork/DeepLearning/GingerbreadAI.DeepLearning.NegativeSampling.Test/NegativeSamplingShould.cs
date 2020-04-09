using System;
using System.Collections.Generic;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Extensions;
using GingerbreadAI.Model.NeuralNetwork.Initialisers;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit;

namespace GingerbreadAI.DeepLearning.NegativeSampling.Test
{
    public class NegativeSamplingShould
    {
        [Fact]
        public void TrainBasicNetworksSortofWell()
        {
            var input = new Layer(5, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var h1 = new Layer(10, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(5, new Layer[] { h1 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);

            output.Initialise(new Random());

            var learningRate = 0.25;
            for (var i = 0; i < 2000; i++)
            {
                output.NegativeSample(0, 0, learningRate, false);
                output.NegativeSample(1, 1, learningRate, false);
                output.NegativeSample(2, 2, learningRate, true);
                output.NegativeSample(3, 3, learningRate, false);
                output.NegativeSample(4, 4, learningRate, false);
            }

            Assert.True(output.GetResult(0, 0) < 0.05);
            Assert.True(output.GetResult(1, 1) < 0.05);
            Assert.True(output.GetResult(2, 2) > 0.95);
            Assert.True(output.GetResult(3, 3) < 0.05);
            Assert.True(output.GetResult(4, 4) < 0.05);
        }

        [Fact]
        public void TrainComplexNetworksSortofWell()
        {
            var input = new Layer(5, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var h1 = new Layer(10, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var h2 = new Layer(10, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var h3 = new Layer(10, new Layer[] { h1, h2 }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(5, new Layer[] { h3 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);

            output.Initialise(new Random());

            var learningRate = 0.25;
            for (var i = 0; i < 2000; i++)
            {
                output.NegativeSample(0, 0, learningRate, false);
                output.NegativeSample(1, 1, learningRate, false);
                output.NegativeSample(2, 2, learningRate, true);
                output.NegativeSample(3, 3, learningRate, false);
                output.NegativeSample(4, 4, learningRate, false);
            }

            Assert.True(output.GetResult(0, 0) < 0.05);
            Assert.True(output.GetResult(1, 1) < 0.05);
            Assert.True(output.GetResult(2, 2) > 0.95);
            Assert.True(output.GetResult(3, 3) < 0.05);
            Assert.True(output.GetResult(4, 4) < 0.05);
        }

        [Fact]
        public void TrainNetworksUsingTheSameTargetSortofWell()
        {
            var input = new Layer(100, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var h1 = new Layer(50, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(100, new Layer[] { h1 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);

            output.Initialise(new Random());

            var learningRate = 0.25;
            for (var i = 0; i < 2000; i++)
            {
                output.NegativeSample(0, 2, learningRate, true);
                output.NegativeSample(1, 2, learningRate, true);
                output.NegativeSample(2, 2, learningRate, false);
                output.NegativeSample(3, 2, learningRate, false);
                output.NegativeSample(4, 2, learningRate, false);
            }

            Assert.True(output.GetResult(0, 2) > 0.95);
            Assert.True(output.GetResult(1, 2) > 0.95);
            Assert.True(output.GetResult(2, 2) < 0.05);
            Assert.True(output.GetResult(3, 2) < 0.05);
            Assert.True(output.GetResult(4, 2) < 0.05);
        }

        [Fact]
        public void TrainNetworksUsingTheSameInputSortofWell()
        {
            var input = new Layer(5, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var h1 = new Layer(50, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(100, new Layer[] { h1 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);

            output.Initialise(new Random());

            var learningRate = 0.25;
            for (var i = 0; i < 2000; i++)
            {
                output.NegativeSample(2, 0, learningRate, true);
                output.NegativeSample(2, 1, learningRate, true);
                output.NegativeSample(2, 2, learningRate, false);
                output.NegativeSample(2, 3, learningRate, false);
                output.NegativeSample(2, 4, learningRate, false);
            }

            Assert.True(output.GetResult(2, 0) > 0.95);
            Assert.True(output.GetResult(2, 1) > 0.95);
            Assert.True(output.GetResult(2, 2) < 0.05);
            Assert.True(output.GetResult(2, 3) < 0.05);
            Assert.True(output.GetResult(2, 4) < 0.05);
        }


        [Fact]
        public void TrainNetworksUsingDifferentInputsAndOutputsSortofWell()
        {
            var input = new Layer(100, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var h1 = new Layer(50, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(100, new Layer[] { h1 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);

            output.Initialise(new Random());

            var learningRate = 0.25;
            for (var i = 0; i < 2000; i++)
            {
                output.NegativeSample(2, 0, learningRate, true);
                output.NegativeSample(2, 1, learningRate, true);
                output.NegativeSample(2, 2, learningRate, false);
                output.NegativeSample(2, 3, learningRate, false);
                output.NegativeSample(2, 4, learningRate, false);

                output.NegativeSample(3, 0, learningRate, false);
                output.NegativeSample(3, 1, learningRate, false);
                output.NegativeSample(3, 2, learningRate, true);
                output.NegativeSample(3, 3, learningRate, true);
                output.NegativeSample(3, 4, learningRate, true);
            }

            Assert.True(output.GetResult(2, 0) > 0.95);
            Assert.True(output.GetResult(2, 1) > 0.95);
            Assert.True(output.GetResult(2, 2) < 0.05);
            Assert.True(output.GetResult(2, 3) < 0.05);
            Assert.True(output.GetResult(2, 4) < 0.05);

            Assert.True(output.GetResult(3, 0) < 0.05);
            Assert.True(output.GetResult(3, 1) < 0.05);
            Assert.True(output.GetResult(3, 2) > 0.95);
            Assert.True(output.GetResult(3, 3) > 0.95);
            Assert.True(output.GetResult(3, 4) > 0.95);
        }

        [Fact]
        public void OnlyChangeRelatedWeights()
        {
            var input = new Layer(10, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var h1 = new Layer(10, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var h2 = new Layer(10, new Layer[] { h1 }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var h3 = new Layer(10, new Layer[] { h1 }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var h4 = new Layer(10, new Layer[] { h2, h3 }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(10, new Layer[] { h4 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);

            output.Initialise(new Random());

            var initialHiddenWeights = new Dictionary<Node, Weight>[h1.Nodes.Length];
            var initialOutputWeights = new Dictionary<Node, Weight>[output.Nodes.Length];
            for (var i = 0; i < h1.Nodes.Length; i++)
            {
                var dict = new Dictionary<Node, Weight>();
                for (var j = 0; j < input.Nodes.Length; j++)
                {
                    dict.Add(input.Nodes[j], new Weight(h1.Nodes[i].Weights[input.Nodes[j]].Value));
                }
                initialHiddenWeights[i] = dict;
            }
            for (var i = 0; i < output.Nodes.Length; i++)
            {
                var dict = new Dictionary<Node, Weight>();
                for (var j = 0; j < h4.Nodes.Length; j++)
                {
                    dict.Add(h4.Nodes[j], new Weight(output.Nodes[i].Weights[h4.Nodes[j]].Value));
                }
                initialOutputWeights[i] = dict;
            }

            var learningRate = 0.25;
            for (var i = 0; i < 2000; i++)
            {
                output.NegativeSample(4, 4, learningRate, true);
            }

            for (var i = 0; i < h1.Nodes.Length; i++)
            {
                for (var j = 0; j < input.Nodes.Length; j++)
                {
                    if (j != 4)
                    {
                        Assert.Equal(initialHiddenWeights[i][input.Nodes[j]].Value, h1.Nodes[i].Weights[input.Nodes[j]].Value);
                    }
                    else
                    {
                        Assert.NotEqual(initialHiddenWeights[i][input.Nodes[j]].Value, h1.Nodes[i].Weights[input.Nodes[j]].Value);
                    }
                }
            }
            for (var i = 0; i < output.Nodes.Length; i++)
            {
                for (var j = 0; j < h4.Nodes.Length; j++)
                {
                    if (i != 4)
                    {
                        Assert.Equal(initialOutputWeights[i][h4.Nodes[j]].Value, output.Nodes[i].Weights[h4.Nodes[j]].Value);
                    }
                    else
                    {
                        Assert.NotEqual(initialOutputWeights[i][h4.Nodes[j]].Value, output.Nodes[i].Weights[h4.Nodes[j]].Value);
                    }
                }
            }
        }
    }
}