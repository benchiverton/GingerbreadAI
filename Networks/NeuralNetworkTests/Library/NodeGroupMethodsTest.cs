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
            var inputGroup = new NodeLayer("Input Group", 1);
            var inner1 = new NodeLayer("Inner 1", 10, new[] { inputGroup });
            var inner2 = new NodeLayer("Inner 2", 10, new[] { inputGroup });
            var inner3 = new NodeLayer("Inner 3", 10, new[] { inner1 });
            var inner4 = new NodeLayer("Inner 4", 10, new[] { inner3 });
            var output = new NodeLayer("Output", 10, new[] { inner2, inner4 });

            var allNodeGroups = NodeLayerMethods.GetAllGroupsInSystem(output);

            Assert.AreEqual(allNodeGroups.Length, 6);
        }
    }
}
