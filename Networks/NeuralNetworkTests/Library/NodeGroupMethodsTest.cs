using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Data;
using NeuralNetwork.Library;

namespace NeuralNetworkTests.Library
{
    [TestClass]
    public class NodeGroupMethodsTest
    {
        [TestMethod]
        public void GetAllGroupsInSystemTest()
        {
            var inputGroup = new NodeGroup("Input Group", 1);
            var inner1 = new NodeGroup("Inner 1", 10, new[] { inputGroup });
            var inner2 = new NodeGroup("Inner 2", 10, new[] { inputGroup });
            var inner3 = new NodeGroup("Inner 3", 10, new[] { inner1 });
            var inner4 = new NodeGroup("Inner 4", 10, new[] { inner3 });
            var output = new NodeGroup("Output", 10, new[] { inner2, inner4 });

            var allNodeGroups = NodeGroupMethods.GetAllGroupsInSystem(output);

            Assert.AreEqual(allNodeGroups.Length, 6);
        }
    }
}
