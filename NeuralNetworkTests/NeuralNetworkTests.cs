using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Networks;
using NeuralNetwork.Nodes;

namespace NeuralNetworkTests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        [TestMethod]
        public void TestInitialising()
        {
            var n = new NodeNetwork();

            // Input layer
            var layer = new NodeLayer("Input", 1);
            n.AddNodeLayer(layer);

            var nodesInner = new[] { new Node(new[] { new[] { 0.2 } }, new[] { 0.7 }) };
            var inner = new NodeLayer("Inner1", nodesInner, new[] { layer });
            n.AddNodeLayer(inner);

            var nodesOuter = new[] { new Node(new[] { new[] { 0.9 } }, new[] { 0.4 }) };
            var outer = new NodeLayer("Inner2", nodesOuter, new[] { inner });
            n.AddNodeLayer(outer);

            // checking that the values calculated in the inner node are correct
            var innerResult = inner.GetResult(new[] { 0.5 });
            Assert.AreEqual(Math.Round(0.68997448112, 4), Math.Round(innerResult[0], 4));

            // checking that the values calculated in the output are correct
            var result = n.GetResult(new[] { 0.5 });
            Assert.AreEqual(Math.Round(0.73516286937, 4), Math.Round(result[0], 4));
        }
    }
}
