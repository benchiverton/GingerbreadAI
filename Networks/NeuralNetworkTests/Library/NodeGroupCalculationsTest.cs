using System;
using System.Diagnostics;
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
            var inputGroup = new NodeGroup("Input Group", 10);
            var outputGroup = new NodeGroup("Output Group", 10, new[] { inputGroup });

            try
            {
                var inputs = new double[5];
                NodeGroupCalculations.GetResult(outputGroup, inputs);
                Assert.Fail("An exception should have been thrown.");
            }
            catch (NodeNetworkException)
            {
                // this is meant to be hit, yay!
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
            // Input group
            var inputGroup = new NodeGroup("Input Group", 1);

            // Hidden group
            var nodesInner = new[]
            {
                new Node
                {
                    Weights = new[] {new[] {0.2}},
                    BiasWeights = new[] {0.7}
                }
            };
            var inner = new NodeGroup("Inner 1", nodesInner, new[] { inputGroup });

            // Output group
            var nodesOuter = new[] { new Node { Weights = new[] { new[] { 0.9 } }, BiasWeights = new[] { 0.4 } } };
            var outer = new NodeGroup("Inner 2", nodesOuter, new[] { inner });

            NodeGroupCalculations.GetResult(outer, new[] { 0.5 });

            // checking that the values calculated in the inner node are correct
            var innerResult = inner.Outputs;
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult[0], 4));

            // checking that the values calculated in the output are correct
            var result = outer.Outputs;
            Assert.AreEqual(Math.Round(0.73516286937, 4), Math.Round(result[0], 4));
        }

        // TODO: need to change the weights and calculate what the relevant outputs should be
        /// <summary>
        ///     Test to verify that if you have more than one group feeding into another group, the correct result is calculated.
        /// </summary>
        [TestMethod]
        public void TestMultipleGroup()
        {
            // Input group
            var inputGroup = new NodeGroup("Input Group", 1);

            // Hidden group 1
            var nodesInner1 = new[] { new Node { Weights = new[] { new[] { 0.2 } }, BiasWeights = new[] { 0.7 } } };
            var inner1 = new NodeGroup("Inner 1", nodesInner1, new[] { inputGroup });

            // Hidden group 2
            var nodesInner2 = new[] { new Node { Weights = new[] { new[] { 0.2 } }, BiasWeights = new[] { 0.7 } } };
            var inner2 = new NodeGroup("Inner 2", nodesInner2, new[] { inputGroup });

            // Output group
            var nodesOut = new[] { new Node { Weights = new[] { new[] { 0.9 }, new[] { 0.9 } }, BiasWeights = new[] { 0.4, 0.4 } } };
            var output = new NodeGroup("Output", nodesOut, new[] { inner1, inner2 });
            NodeGroupCalculations.GetResult(output, new[] { 0.5 });

            // checking that the values calculated in the inner1 node are correct
            var innerResult1 = inner1.Outputs;
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult1[0], 4));

            // checking that the values calculated in the inner2 node are correct
            var innerResult2 = inner2.Outputs;
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult2[0], 4));

            // checking that the values calculated in the output are correct
            var result = output.Outputs;
            Assert.AreEqual(Math.Round(0.8851320938059, 4), Math.Round(result[0], 4));
        }

        // TODO: add a test method which tests ~3 nodes on one group
        /// <summary>
        ///     Test to verify that if we have more than one node on a group, the correct output is calculated.
        /// </summary>
        [TestMethod]
        public void TestMultiNodeGroup()
        {
            var inputGroup = new NodeGroup("Input Group", 2);
            var nodesInner = new[] {
                new Node { Weights = new[] { new[] { 0.02, 0.07 }, }, BiasWeights = new[] { (double) 0 } },
                new Node { Weights = new[] { new[] { 0.03, 0.11 }, }, BiasWeights = new[] { (double) 0 } },
                new Node { Weights = new[] { new[] { 0.05, 0.13 }, }, BiasWeights = new[] { (double) 0 } },
            };
            var inner = new NodeGroup("Inner 1", nodesInner, new[] { inputGroup });
            var nodesOutput = new[] {
                new Node { Weights = new[] { new[] { 0.17, 0.23, 0.31 }, }, BiasWeights = new[] { (double) 0 } },
                new Node { Weights = new[] { new[] { 0.19, 0.29, 0.37 }, }, BiasWeights = new[] { (double) 0 } },
            };
            var output = new NodeGroup("Inner 1", nodesOutput, new[] { inner });

            NodeGroupCalculations.GetResult(output, new[] { 41.0, 43.0 });

            Assert.AreEqual(Math.Round(inner.Outputs[0], 8), Math.Round(0.978751677288986, 8));
            Assert.AreEqual(Math.Round(inner.Outputs[1], 8), Math.Round(0.99742672684619, 8));
            Assert.AreEqual(Math.Round(inner.Outputs[2], 8), Math.Round(0.99951940263283, 8));
            Assert.AreEqual(Math.Round(output.Outputs[0], 8), Math.Round(0.669438581764625, 8));
            Assert.AreEqual(Math.Round(output.Outputs[1], 8), Math.Round(0.699525372246435, 8));
        }

        /// <summary>
        ///     Test to check the efficiency of the GetResult() method (the time taken should be as small as possible).
        /// </summary>
        [TestMethod]
        public void TestGetResultEfficiency()
        {
            const int calcCount = 5000;

            var group = new NodeGroup("Input", 20);
            var inner1 = new NodeGroup("Inner1", 100, new[] { group });
            var inner2 = new NodeGroup("Inner2", 100, new[] { group });
            var output = new NodeGroup("Output", 20, new[] { inner1, inner2 });

            Initialiser.Initialise(new Random(), output);

            var inputs = new double[20];
            for (var i = 0; i < inputs.Length; i++)
                inputs[i] = 0.5;

            var stopWatch = new Stopwatch();
            stopWatch.Start();

            // Gets the result every time this loop iterates
            for (var i = 0; i < calcCount; i++)
                NodeGroupCalculations.GetResult(output, inputs);

            stopWatch.Stop();
            Console.WriteLine($"{calcCount} calculations took {stopWatch.ElapsedMilliseconds}ms.");
        }
    }
}