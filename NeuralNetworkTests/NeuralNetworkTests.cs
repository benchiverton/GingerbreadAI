using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Exceptions;
using NeuralNetwork.Networks;
using NeuralNetwork.Nodes;

namespace NeuralNetworkTests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        /// <summary>
        /// Tests to see if the GetResult() method produces the correct value.
        /// Network structure being tested:
        /// Input: 1 node,
        /// Hidden: 1 layer (with 1 node),
        /// Output: 1 node.
        /// </summary>
        [TestMethod]
        public void TestBasicNetworkResult()
        {
            var n = new NodeNetwork();

            // Input layer
            var inputLayer = new NodeLayer("Input Layer", 1);
            n.AddNodeLayer(inputLayer);

            // Hidden layer
            var nodesInner = new[] { new Node(new[] { new[] { 0.2 } }, new[] { 0.7 }) };
            var inner = new NodeLayer("Inner 1", nodesInner, new[] { inputLayer });
            n.AddNodeLayer(inner);

            // Output layer
            var nodesOuter = new[] { new Node(new[] { new[] { 0.9 } }, new[] { 0.4 }) };
            var outer = new NodeLayer("Inner 2", nodesOuter, new[] { inner });
            n.AddNodeLayer(outer);

            // checking that the values calculated in the inner node are correct
            var innerResult = inner.GetResult(new[] { 0.5 });
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult[0], 4));

            // checking that the values calculated in the output are correct
            var result = n.GetResult(new[] { 0.5 });
            Assert.AreEqual(Math.Round(0.73516286937, 4), Math.Round(result[0], 4));
        }

        /// <summary>
        /// Test to verify that if you enter an incorrect input array throws the correct error.
        /// </summary>
        [TestMethod]
        public void TestGetResultIncorrectInput()
        {
            var n = new NodeNetwork();

            // Input layer
            var inputLayer = new NodeLayer("Input Layer", 10);
            n.AddNodeLayer(inputLayer);

            var outputLayer = new NodeLayer("Output Layer", 10, new [] {inputLayer});
            n.AddNodeLayer(outputLayer);

            try
            {
                var inputs = new double[5];
                n.GetResult(inputs);
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
        /// Test to check the efficiency of the GetResult() method (the time taken should be as small as possible).
        /// </summary>
        [TestMethod]
        public void TestGetResultEfficiency()
        {
            var n = new NodeNetwork();

            var layer = new NodeLayer("Input", 20);
            n.AddNodeLayer(layer);

            var inner1 = new NodeLayer("Inner1", 100, new[] { layer });
            n.AddNodeLayer(inner1);

            var inner2 = new NodeLayer("Inner2", 100, new[] { layer });
            n.AddNodeLayer(inner2);

            var output = new NodeLayer("Output", 20, new[] { inner1, inner2 });
            n.AddNodeLayer(output);

            n.Initialise(new Random());

            var inputs = new double[20];
            for(var i=0; i<inputs.Length; i++)
            {
                inputs[i] = 0.5;
            }

            for (var i = 0; i < 5000; i++)
            {
                n.GetResult(inputs);
            }
        }
    }
}
