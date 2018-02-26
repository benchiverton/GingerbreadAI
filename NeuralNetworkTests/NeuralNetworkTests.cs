using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Data;
using NeuralNetwork.Exceptions;
using NeuralNetwork.Library;

namespace NeuralNetworkTests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        /// <summary>
        ///     Test to verify that if you enter an input array of an incorrect magnitude, then the correct erro is thrown.
        /// </summary>
        [TestMethod]
        public void TestGetResultIncorrectInput()
        {
            var n = new NodeNetwork();

            // Input layer
            var inputLayer = new NodeLayer("Input Layer", 10);
            NodeNetworkCalculations.AddNodeLayer(inputLayer, n);

            var outputLayer = new NodeLayer("Output Layer", 10, new[] {inputLayer});
            NodeNetworkCalculations.AddNodeLayer(outputLayer, n);

            try
            {
                var inputs = new double[5];
                NodeNetworkCalculations.GetResult(inputs, n);
                Assert.Fail("An exception should have been thrown.");
            }
            catch (NodeNetworkException ex)
            {
                Assert.AreEqual("Please enter the correct amount of inputs for your network.", ex.Message);
            }
            catch (Exception)
            {
                Assert.Fail("The Exception thrown was not of type NodeNetworkException.");
            }
        }

        /// <summary>
        ///     Tests to see if the GetResult() method produces the correct value with a specific structure.
        ///     Network structure being tested:
        ///     Input: 1 node,
        ///     Hidden: 1 layer (with 1 node),
        ///     Output: 1 node.
        /// </summary>
        [TestMethod]
        public void TestBasicNetworkResult()
        {
            var n = new NodeNetwork();

            // Input layer
            var inputLayer = new NodeLayer("Input Layer", 1);
            NodeNetworkCalculations.AddNodeLayer(inputLayer, n);

            // Hidden layer
            var nodesInner = new[]
            {
                new Node
                {
                    Weights = new[] {new[] {0.2}},
                    BiasWeights = new[] {0.7}
                }
            };
            var inner = new NodeLayer("Inner 1", nodesInner, new[] {inputLayer});
            NodeNetworkCalculations.AddNodeLayer(inner, n);

            // Output layer
            var nodesOuter = new[] {new Node {Weights = new[] {new[] {0.9}}, BiasWeights = new[] {0.4}}};
            var outer = new NodeLayer("Inner 2", nodesOuter, new[] {inner});
            NodeNetworkCalculations.AddNodeLayer(outer, n);

            // checking that the values calculated in the inner node are correct
            var innerResult = NodeLayerCalculations.GetResult(new[] {0.5}, inner);
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult[0], 4));

            // checking that the values calculated in the output are correct
            var result = NodeNetworkCalculations.GetResult(new[] {0.5}, n);
            Assert.AreEqual(Math.Round(0.73516286937, 4), Math.Round(result[0], 4));
        }

        // TODO: need to change the weights and calculate what the relevant outputs should be
        /// <summary>
        ///     Test to verify that if you have more than one layer feeding into another layer, the correct result is calculated.
        /// </summary>
        [TestMethod]
        public void TestMultipleLayer()
        {
            var n = new NodeNetwork();

            // Input layer
            var inputLayer = new NodeLayer("Input Layer", 1);
            NodeNetworkCalculations.AddNodeLayer(inputLayer, n);

            // Hidden layer 1
            var nodesInner1 = new[] {new Node{Weights = new[] {new[] {0.2}}, BiasWeights = new[] {0.7}}};
            var inner1 = new NodeLayer("Inner 1", nodesInner1, new[] {inputLayer});
            NodeNetworkCalculations.AddNodeLayer(inner1, n);

            // Hidden layer 2
            var nodesInner2 = new[] {new Node{Weights = new[] {new[] {0.2}}, BiasWeights = new[] {0.7}}};
            var inner2 = new NodeLayer("Inner 2", nodesInner2, new[] {inputLayer});
            NodeNetworkCalculations.AddNodeLayer(inner2, n);

            // Output layer
            var nodesOut = new[] {new Node{Weights = new[] {new[] {0.9}, new[] {0.9}}, BiasWeights = new[] {0.4, 0.4}}};
            var output = new NodeLayer("Output", nodesOut, new[] {inner1, inner2});
            NodeNetworkCalculations.AddNodeLayer(output, n);

            // checking that the values calculated in the inner1 node are correct
            var innerResult1 = NodeLayerCalculations.GetResult(new[] {0.5}, inner1);
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult1[0], 4));

            // checking that the values calculated in the inner2 node are correct
            var innerResult2 = NodeLayerCalculations.GetResult(new[] {0.5}, inner2);
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult2[0], 4));

            // checking that the values calculated in the output are correct
            var result = NodeNetworkCalculations.GetResult(new[] {0.5}, n);
            Assert.AreEqual(Math.Round(0.8851320938059, 4), Math.Round(result[0], 4));
        }

        // TODO: add a test method which tests ~3 nodes on one layer
        /// <summary>
        ///     Test to verify that if we have more than one node on a layer, the correct output is calculated.
        /// </summary>
        [TestMethod]
        public void TestMultiNodeLayer()
        {
            throw new NotImplementedException();
        }

        // TODO: add a test method which tests multiple input layers
        /// <summary>
        ///     Test to varify that if we have more than one input later, the correct output is calculated
        /// </summary>
        [TestMethod]
        public void TestMultipleInputs()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///     Test to check the efficiency of the GetResult() method (the time taken should be as small as possible).
        /// </summary>
        [TestMethod]
        public void TestGetResultEfficiency()
        {
            var n = new NodeNetwork();

            var layer = new NodeLayer("Input", 20);
            NodeNetworkCalculations.AddNodeLayer(layer, n);

            var inner1 = new NodeLayer("Inner1", 100, new[] {layer});
            NodeNetworkCalculations.AddNodeLayer(inner1, n);

            var inner2 = new NodeLayer("Inner2", 100, new[] {layer});
            NodeNetworkCalculations.AddNodeLayer(inner2, n);

            var output = new NodeLayer("Output", 20, new[] {inner1, inner2});
            NodeNetworkCalculations.AddNodeLayer(output, n);

            Initialiser.Initialise(new Random(), n);

            var inputs = new double[20];
            for (var i = 0; i < inputs.Length; i++)
                inputs[i] = 0.5;

            // Gets the result every time this loop iterates
            for (var i = 0; i < 5000; i++)
                NodeNetworkCalculations.GetResult(inputs, n);
        }
    }
}