using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;

namespace Model.NeuralNetwork.Test.Models
{
    using System;
    using System.Collections.Generic;
    using Xunit;
    using Xunit.Abstractions;

    public class LayerShould
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public LayerShould(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void ThrowAnExceptionWhenInputIsInvalid()
        {
            // Input group
            var inputGroup = new Layer(10, new Layer[0], ActivationFunctionType.Sigmoid, InitialisationFunctionType.None);
            var outputGroup = new Layer(10, new[] { inputGroup }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.None);

            var inputs = new double[5];
            Assert.Throws<ArgumentException>(() => outputGroup.CalculateOutputs(inputs));
        }

        [Fact]
        public void CalculateBasicResultCorrectly()
        {
            // Input group
            var inputLayer = new Layer(1, new Layer[0], ActivationFunctionType.Sigmoid, InitialisationFunctionType.None);
            // Hidden group
            var innerNodeInfo = new Dictionary<Layer, (double[], double)>
            {
                { inputLayer, (new[] { 0.2 }, 0.7) }
            };
            var innerNode = GenerateWeightedNode(innerNodeInfo);
            var innerLayer = new Layer
            {
                Nodes = new[] { innerNode },
                PreviousLayers = new[] { inputLayer },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };
            // Output group
            var outputNodeInfo = new Dictionary<Layer, (double[], double)>
            {
                { innerLayer, (new[] { 0.9 }, 0.4) }
            };
            var outputNode = GenerateWeightedNode(outputNodeInfo);
            var outputLayer = new Layer
            {
                Nodes = new[] { outputNode },
                PreviousLayers = new[] { innerLayer },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };

            outputLayer.CalculateOutputs(new[] { 0.5 });

            // checking that the values calculated in the inner node are correct
            var innerResult = innerLayer.Nodes[0].Output;
            Assert.Equal(Math.Round(0.68997448112, 4), Math.Round(innerResult, 4));
            // checking that the values calculated in the output are correct
            var result = outputLayer.Nodes[0].Output;
            Assert.Equal(Math.Round(0.73516286937, 4), Math.Round(result, 4));
        }

        [Fact]
        public void CalculateIndexedResultCorrectly()
        {
            // Input group
            var inputLayer = new Layer(5, new Layer[0], ActivationFunctionType.Sigmoid, InitialisationFunctionType.None);
            // Hidden group
            var innerNodeInfo = new Dictionary<Layer, (double[], double)>
            {
                { inputLayer, (new[] { 1d, 1d, 0.2, 1d, 1d }, 0.7) }
            };
            var innerNode = GenerateWeightedNode(innerNodeInfo);
            var innerLayer = new Layer
            {
                Nodes = new[] { innerNode },
                PreviousLayers = new[] { inputLayer },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };
            // Output group
            var outputNodeInfo1 = new Dictionary<Layer, (double[], double)>
            {
                { innerLayer, (new[] { 1d }, 0.4) }
            };
            var outputNodeInfo2 = new Dictionary<Layer, (double[], double)>
            {
                { innerLayer, (new[] { 0.9 }, 0.4) }
            };
            var outputNodeInfo3 = new Dictionary<Layer, (double[], double)>
            {
                { innerLayer, (new[] { 1d }, 0.4) }
            };
            var outputNode1 = GenerateWeightedNode(outputNodeInfo1);
            var outputNode2 = GenerateWeightedNode(outputNodeInfo2);
            var outputNode3 = GenerateWeightedNode(outputNodeInfo3);
            var outputLayer = new Layer
            {
                Nodes = new[] { outputNode1, outputNode2, outputNode3 },
                PreviousLayers = new[] { innerLayer },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };

            outputLayer.CalculateIndexedOutput(2, 1, 0.5);

            // checking that the values calculated in the inner node are correct
            var innerResult = innerLayer.Nodes[0].Output;
            Assert.Equal(Math.Round(0.68997448112, 4), Math.Round(innerResult, 4));
            // checking that the values calculated in the output are correct
            var result = outputLayer.Nodes[1].Output;
            Assert.Equal(Math.Round(0.73516286937, 4), Math.Round(result, 4));
        }

        [Fact]
        public void CalculateMultipleGroupsResultCorrectly()
        {
            // Input group
            var inputLayer = new Layer(1, new Layer[0], ActivationFunctionType.Sigmoid, InitialisationFunctionType.None);
            // Hidden group 1
            // Hidden group
            var innerNodeInfo1 = new Dictionary<Layer, (double[], double)>
            {
                { inputLayer, (new[] { 0.2 }, 0.7) }
            };
            var innerNode1 = GenerateWeightedNode(innerNodeInfo1);
            var innerLayer1 = new Layer
            {
                Nodes = new[] { innerNode1 },
                PreviousLayers = new[] { inputLayer },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };
            // Hidden group 2
            // Hidden group
            var innerNodeInfo2 = new Dictionary<Layer, (double[], double)>
            {
                { inputLayer, (new[] { 0.2 }, 0.7) }
            };
            var innerNode2 = GenerateWeightedNode(innerNodeInfo2);
            var innerLayer2 = new Layer
            {
                Nodes = new[] { innerNode2 },
                PreviousLayers = new[] { inputLayer },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };
            // Output group
            var outputNodeInfo = new Dictionary<Layer, (double[], double)>
            {
                { innerLayer1, (new[] { 0.9 }, 0.4) },
                { innerLayer2, (new[] { 0.9 }, 0.4) }
            };
            var outputNode = GenerateWeightedNode(outputNodeInfo);
            var outputLayer = new Layer
            {
                Nodes = new[] { outputNode },
                PreviousLayers = new[] { innerLayer1, innerLayer2 },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };

            outputLayer.CalculateOutputs(new[] { 0.5 });

            // checking that the values calculated in the inner1 node are correct
            var innerResult1 = innerLayer1.Nodes[0].Output;
            Assert.Equal(Math.Round(0.68997448112, 4), Math.Round(innerResult1, 4));
            // checking that the values calculated in the inner2 node are correct
            var innerResult2 = innerLayer2.Nodes[0].Output;
            Assert.Equal(Math.Round(0.68997448112, 4), Math.Round(innerResult2, 4));
            // checking that the values calculated in the output are correct
            var result = outputLayer.Nodes[0].Output;
            Assert.Equal(Math.Round(0.8851320938059, 4), Math.Round(result, 4));
        }

        [Fact]
        public void CalculateMultipleNodesResultCorrectly()
        {
            var inputLayer = new Layer(2, new Layer[0], ActivationFunctionType.Sigmoid, InitialisationFunctionType.None);
            var innerNode1Info = new Dictionary<Layer, (double[], double)>
            {
                { inputLayer, (new[] { 0.02, 0.07 }, 0) }
            };
            var innerNode1 = GenerateWeightedNode(innerNode1Info);
            var innerNode2Info = new Dictionary<Layer, (double[], double)>
            {
                { inputLayer, (new[] { 0.03, 0.11 }, 0) }
            };
            var innerNode2 = GenerateWeightedNode(innerNode2Info);
            var innerNode3Info = new Dictionary<Layer, (double[], double)>
            {
                { inputLayer, (new[] { 0.05, 0.13 }, 0) }
            };
            var innerNode3 = GenerateWeightedNode(innerNode3Info);
            var innerLayer = new Layer
            {
                Nodes = new[] { innerNode1, innerNode2, innerNode3 },
                PreviousLayers = new[] { inputLayer },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };
            var outerNode1Info = new Dictionary<Layer, (double[], double)>
            {
                { innerLayer, (new[] { 0.17, 0.23, 0.31 }, 0) }
            };
            var outerNode1 = GenerateWeightedNode(outerNode1Info);
            var outerNode2Info = new Dictionary<Layer, (double[], double)>
            {
                { innerLayer, (new[] { 0.19, 0.29, 0.37 }, 0) }
            };
            var outerNode2 = GenerateWeightedNode(outerNode2Info);
            var output = new Layer
            {
                Nodes = new[] { outerNode1, outerNode2 },
                PreviousLayers = new[] { innerLayer },
                ActivationFunctionType = ActivationFunctionType.Sigmoid
            };

            output.CalculateOutputs(new[] { 41.0, 43.0 });

            Assert.Equal(Math.Round(innerLayer.Nodes[0].Output, 8), Math.Round(0.978751677288986, 8));
            Assert.Equal(Math.Round(innerLayer.Nodes[1].Output, 8), Math.Round(0.99742672684619, 8));
            Assert.Equal(Math.Round(innerLayer.Nodes[2].Output, 8), Math.Round(0.99951940263283, 8));
            Assert.Equal(Math.Round(output.Nodes[0].Output, 8), Math.Round(0.669438581764625, 8));
            Assert.Equal(Math.Round(output.Nodes[1].Output, 8), Math.Round(0.699525372246435, 8));
        }

        private Node GenerateWeightedNode(Dictionary<Layer, (double[] layerWeights, double biasWeight)> nodeInformation)
        {
            var weights = new Dictionary<Node, Weight>();
            var biasWeights = new Dictionary<Layer, Weight>();

            foreach (var nodeLayer in nodeInformation.Keys)
            {
                var data = nodeInformation[nodeLayer];

                if (data.layerWeights.Length != nodeLayer.Nodes.Length)
                {
                    throw new Exception("Incorrect amount of weights supplied");
                }

                for (var i = 0; i < data.layerWeights.Length; i++)
                {
                    weights.Add(nodeLayer.Nodes[i], new Weight(data.layerWeights[i]));
                }

                biasWeights.Add(nodeLayer, new Weight(data.biasWeight));
            }

            return new Node
            {
                Weights = weights,
                BiasWeights = biasWeights
            };
        }
    }
}