using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Data;
using NeuralNetwork.Exceptions;
using NeuralNetwork.Library;

namespace NeuralNetworkTests.Library
{
    [TestClass]
    public class NodeGroupCalculationsTest
    {
        /// <summary>
        ///     Test to verify that if you enter an input array of an incorrect magnitude, then the correct erro is thrown.
        /// </summary>
        [TestMethod]
        public void TestGetResultIncorrectInput()
        {
            // Input group
            var inputGroup = new NodeLayer("Input Group", 10, new NodeLayer[0]);
            var outputGroup = new NodeLayer("Output Group", 10, new[] { inputGroup });

            try
            {
                var inputs = new double[5];
                var nodeLayerLogic = new NodeLayerLogic
                {
                    OutputLayer = outputGroup
                };
                nodeLayerLogic.PopulateResults(inputs);
                Assert.Fail("An exception should have been thrown.");
            }
            catch (NodeNetworkException)
            {
                // this is meant to be hit, yay!
            }
        }

        [TestMethod]
        public void TestBasicNetworkResult()
        {
            // Input group
            var inputLayer = new NodeLayer("Input Layer", 1, new NodeLayer[0]);
            // Hidden group
            var innerNodeInfo = new Dictionary<NodeLayer, (double[], double)>
            {
                { inputLayer, (new[] { 0.2 }, 0.7) }
            };
            var innerNode = GenerateWeightedNode(innerNodeInfo);
            var innerLayer = new NodeLayer
            {
                Name = "Inner Layer",
                Nodes = new[] { innerNode },
                PreviousLayers = new[] { inputLayer }
            };
            // Output group
            var outputNodeInfo = new Dictionary<NodeLayer, (double[], double)>
            {
                { innerLayer, (new[] { 0.9 }, 0.4) }
            };
            var outputNode = GenerateWeightedNode(outputNodeInfo);
            var outputLayer = new NodeLayer
            {
                Name = "Output Layer",
                Nodes = new[] { outputNode },
                PreviousLayers = new[] { innerLayer }
            };

            var nodeLayerLogic = new NodeLayerLogic { OutputLayer = outputLayer };
            nodeLayerLogic.PopulateResults(new[] { 0.5 });

            // checking that the values calculated in the inner node are correct
            var innerResult = innerLayer.Nodes[0].Output;
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult, 4));
            // checking that the values calculated in the output are correct
            var result = outputLayer.Nodes[0].Output;
            Assert.AreEqual(Math.Round(0.73516286937, 4), Math.Round(result, 4));
        }

        [TestMethod]
        public void TestMultipleGroup()
        {
            // Input group
            var inputLayer = new NodeLayer("Input Layer", 1, new NodeLayer[0]);
            // Hidden group 1
            // Hidden group
            var innerNodeInfo1 = new Dictionary<NodeLayer, (double[], double)>
            {
                { inputLayer, (new[] { 0.2 }, 0.7) }
            };
            var innerNode1 = GenerateWeightedNode(innerNodeInfo1);
            var innerLayer1 = new NodeLayer
            {
                Name = "Inner Layer 1",
                Nodes = new[] { innerNode1 },
                PreviousLayers = new[] { inputLayer }
            };
            // Hidden group 2
            // Hidden group
            var innerNodeInfo2 = new Dictionary<NodeLayer, (double[], double)>
            {
                { inputLayer, (new[] { 0.2 }, 0.7) }
            };
            var innerNode2 = GenerateWeightedNode(innerNodeInfo2);
            var innerLayer2 = new NodeLayer
            {
                Name = "Inner Layer 2",
                Nodes = new[] { innerNode2 },
                PreviousLayers = new[] { inputLayer }
            };
            // Output group
            var outputNodeInfo = new Dictionary<NodeLayer, (double[], double)>
            {
                { innerLayer1, (new[] { 0.9 }, 0.4) },
                { innerLayer2, (new[] { 0.9 }, 0.4) }
            };
            var outputNode = GenerateWeightedNode(outputNodeInfo);
            var outputLayer = new NodeLayer
            {
                Name = "Output Layer",
                Nodes = new[] { outputNode },
                PreviousLayers = new[] { innerLayer1, innerLayer2 }
            };

            var nodeLayerLogic = new NodeLayerLogic { OutputLayer = outputLayer };
            nodeLayerLogic.PopulateResults(new[] { 0.5 });

            // checking that the values calculated in the inner1 node are correct
            var innerResult1 = innerLayer1.Nodes[0].Output;
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult1, 4));
            // checking that the values calculated in the inner2 node are correct
            var innerResult2 = innerLayer2.Nodes[0].Output;
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult2, 4));
            // checking that the values calculated in the output are correct
            var result = outputLayer.Nodes[0].Output;
            Assert.AreEqual(Math.Round(0.8851320938059, 4), Math.Round(result, 4));
        }

        [TestMethod]
        public void TestMultiNodeGroup()
        {
            var inputLayer = new NodeLayer("Input Group", 2, new NodeLayer[0]);
            var innerNode1Info = new Dictionary<NodeLayer, (double[], double)>
            {
                { inputLayer, (new[] { 0.02, 0.07 }, 0) }
            };
            var innerNode1 = GenerateWeightedNode(innerNode1Info);
            var innerNode2Info = new Dictionary<NodeLayer, (double[], double)>
            {
                { inputLayer, (new[] { 0.03, 0.11 }, 0) }
            };
            var innerNode2 = GenerateWeightedNode(innerNode2Info);
            var innerNode3Info = new Dictionary<NodeLayer, (double[], double)>
            {
                { inputLayer, (new[] { 0.05, 0.13 }, 0) }
            };
            var innerNode3 = GenerateWeightedNode(innerNode3Info);
            var innerLayer = new NodeLayer
            {
                Name = "Inner 1",
                Nodes = new[] { innerNode1, innerNode2, innerNode3 },
                PreviousLayers = new[] { inputLayer }
            };
            var outerNode1Info = new Dictionary<NodeLayer, (double[], double)>
            {
                { innerLayer, (new[] { 0.17, 0.23, 0.31 }, 0) }
            };
            var outerNode1 = GenerateWeightedNode(outerNode1Info);
            var outerNode2Info = new Dictionary<NodeLayer, (double[], double)>
            {
                { innerLayer, (new[] { 0.19, 0.29, 0.37 }, 0) }
            };
            var outerNode2 = GenerateWeightedNode(outerNode2Info);
            var output = new NodeLayer
            {
                Name = "Inner 1",
                Nodes = new[] { outerNode1, outerNode2 },
                PreviousLayers = new[] { innerLayer }
            };

            var nodeLayerLogic = new NodeLayerLogic
            {
                OutputLayer = output
            };
            nodeLayerLogic.PopulateResults(new[] { 41.0, 43.0 });

            Assert.AreEqual(Math.Round(innerLayer.Nodes[0].Output, 8), Math.Round(0.978751677288986, 8));
            Assert.AreEqual(Math.Round(innerLayer.Nodes[1].Output, 8), Math.Round(0.99742672684619, 8));
            Assert.AreEqual(Math.Round(innerLayer.Nodes[2].Output, 8), Math.Round(0.99951940263283, 8));
            Assert.AreEqual(Math.Round(output.Nodes[0].Output, 8), Math.Round(0.669438581764625, 8));
            Assert.AreEqual(Math.Round(output.Nodes[1].Output, 8), Math.Round(0.699525372246435, 8));
        }

        /// <summary>
        ///     Test to check the efficiency of the GetResult() method (the time taken should be as small as possible).
        /// </summary>
        [TestMethod]
        public void TestGetResultEfficiency()
        {
            const int calcCount = 5000;
            var group = new NodeLayer("Input", 20, new NodeLayer[0]);
            var inner1 = new NodeLayer("Inner1", 100, new[] { group });
            var inner2 = new NodeLayer("Inner2", 100, new[] { group });
            var output = new NodeLayer("Output", 20, new[] { inner1, inner2 });
            Initialiser.Initialise(new Random(), output);
            var inputs = new double[20];
            for (var i = 0; i < inputs.Length; i++)
                inputs[i] = 0.5;
            var stopWatch = new Stopwatch();

            stopWatch.Start();
            var nodeLayerLogic = new NodeLayerLogic
            {
                OutputLayer = output
            };
            // Gets the result every time this loop iterates
            for (var i = 0; i < calcCount; i++)
            {
                nodeLayerLogic.PopulateResults(inputs);
            }
            stopWatch.Stop();

            Console.WriteLine($"{calcCount} calculations took {stopWatch.ElapsedMilliseconds}ms.");
        }

        private Node GenerateWeightedNode(Dictionary<NodeLayer, (double[] layerWeights, double biasWeight)> nodeInformation)
        {
            var w8s = new Dictionary<NodeLayer, Dictionary<Node, double>>();
            var biasw8s = new Dictionary<NodeLayer, double>();

            foreach (var nodeLayer in nodeInformation.Keys)
            {
                var data = nodeInformation[nodeLayer];

                if (data.layerWeights.Length != nodeLayer.Nodes.Length)
                {
                    throw new Exception("Incorrect amount of weights supplied");
                }

                var lyrw8s = new Dictionary<Node, double>();
                for (int i = 0; i < data.layerWeights.Length; i++)
                {
                    lyrw8s.Add(nodeLayer.Nodes[i], data.layerWeights[i]);
                }

                w8s.Add(nodeLayer, lyrw8s);
                biasw8s.Add(nodeLayer, data.biasWeight);
            }

            return new Node
            {
                Weights = w8s,
                BiasWeights = biasw8s,
                Output = 0
            };
        }
    }
}