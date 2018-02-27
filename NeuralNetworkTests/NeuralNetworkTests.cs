using System;
using System.Diagnostics;
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

            // Input group
            var inputGroup = new NodeGroup("Input Group", 10);
            NodeNetworkCalculations.AddNodeGroup(inputGroup, n);

            var outputGroup = new NodeGroup("Output Group", 10, new[] {inputGroup});
            NodeNetworkCalculations.AddNodeGroup(outputGroup, n);

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
        ///     Hidden: 1 group (with 1 node),
        ///     Output: 1 node.
        /// </summary>
        [TestMethod]
        public void TestBasicNetworkResult()
        {
            var n = new NodeNetwork();

            // Input group
            var inputGroup = new NodeGroup("Input Group", 1);
            NodeNetworkCalculations.AddNodeGroup(inputGroup, n);

            // Hidden group
            var nodesInner = new[]
            {
                new Node
                {
                    Weights = new[] {new[] {0.2}},
                    BiasWeights = new[] {0.7}
                }
            };
            var inner = new NodeGroup("Inner 1", nodesInner, new[] {inputGroup});
            NodeNetworkCalculations.AddNodeGroup(inner, n);

            // Output group
            var nodesOuter = new[] {new Node {Weights = new[] {new[] {0.9}}, BiasWeights = new[] {0.4}}};
            var outer = new NodeGroup("Inner 2", nodesOuter, new[] {inner});
            NodeNetworkCalculations.AddNodeGroup(outer, n);

            // checking that the values calculated in the inner node are correct
            var innerResult = NodeGroupCalculations.GetResult(new[] {0.5}, inner);
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult[0], 4));

            // checking that the values calculated in the output are correct
            var result = NodeNetworkCalculations.GetResult(new[] {0.5}, n);
            Assert.AreEqual(Math.Round(0.73516286937, 4), Math.Round(result[0], 4));
        }

        // TODO: need to change the weights and calculate what the relevant outputs should be
        /// <summary>
        ///     Test to verify that if you have more than one group feeding into another group, the correct result is calculated.
        /// </summary>
        [TestMethod]
        public void TestMultipleGroup()
        {
            var n = new NodeNetwork();

            // Input group
            var inputGroup = new NodeGroup("Input Group", 1);
            NodeNetworkCalculations.AddNodeGroup(inputGroup, n);

            // Hidden group 1
            var nodesInner1 = new[] {new Node{Weights = new[] {new[] {0.2}}, BiasWeights = new[] {0.7}}};
            var inner1 = new NodeGroup("Inner 1", nodesInner1, new[] {inputGroup});
            NodeNetworkCalculations.AddNodeGroup(inner1, n);

            // Hidden group 2
            var nodesInner2 = new[] {new Node{Weights = new[] {new[] {0.2}}, BiasWeights = new[] {0.7}}};
            var inner2 = new NodeGroup("Inner 2", nodesInner2, new[] {inputGroup});
            NodeNetworkCalculations.AddNodeGroup(inner2, n);

            // Output group
            var nodesOut = new[] {new Node{Weights = new[] {new[] {0.9}, new[] {0.9}}, BiasWeights = new[] {0.4, 0.4}}};
            var output = new NodeGroup("Output", nodesOut, new[] {inner1, inner2});
            NodeNetworkCalculations.AddNodeGroup(output, n);

            // checking that the values calculated in the inner1 node are correct
            var innerResult1 = NodeGroupCalculations.GetResult(new[] {0.5}, inner1);
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult1[0], 4));

            // checking that the values calculated in the inner2 node are correct
            var innerResult2 = NodeGroupCalculations.GetResult(new[] {0.5}, inner2);
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult2[0], 4));

            // checking that the values calculated in the output are correct
            var result = NodeNetworkCalculations.GetResult(new[] {0.5}, n);
            Assert.AreEqual(Math.Round(0.8851320938059, 4), Math.Round(result[0], 4));
        }

        // TODO: add a test method which tests ~3 nodes on one group
        /// <summary>
        ///     Test to verify that if we have more than one node on a group, the correct output is calculated.
        /// </summary>
        [TestMethod]
        public void TestMultiNodeGroup()
        {
            throw new NotImplementedException();
        }

        // TODO: add a test method which tests multiple input groups
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
            const int calcCount = 5000;
            var n = new NodeNetwork();

            var group = new NodeGroup("Input", 20);
            NodeNetworkCalculations.AddNodeGroup(group, n);

            var inner1 = new NodeGroup("Inner1", 100, new[] {group});
            NodeNetworkCalculations.AddNodeGroup(inner1, n);

            var inner2 = new NodeGroup("Inner2", 100, new[] {group});
            NodeNetworkCalculations.AddNodeGroup(inner2, n);

            var output = new NodeGroup("Output", 20, new[] {inner1, inner2});
            NodeNetworkCalculations.AddNodeGroup(output, n);

            Initialiser.Initialise(new Random(), n);

            var inputs = new double[20];
            for (var i = 0; i < inputs.Length; i++)
                inputs[i] = 0.5;

            var stopWatch = new Stopwatch();
            stopWatch.Start();

            // Gets the result every time this loop iterates
            for (var i = 0; i < calcCount; i++)
                NodeNetworkCalculations.GetResult(inputs, n);

            stopWatch.Stop();
            Console.WriteLine($"{calcCount} calculations took {stopWatch.ElapsedMilliseconds}ms.");
        }
    }
}