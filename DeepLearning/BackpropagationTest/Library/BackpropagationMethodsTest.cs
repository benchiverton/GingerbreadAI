using Backpropagation.Library;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Data;
using System.Linq;

namespace BackpropagationTest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void GenerateBackpropagationGroupsDataTest()
        {
            var inputGroup = new NodeLayer("Input Group", 1);
            var inner1 = new NodeLayer("Inner 1", 10, new[] { inputGroup });
            var inner2 = new NodeLayer("Inner 2", 10, new[] { inputGroup });
            var inner3 = new NodeLayer("Inner 3", 10, new[] { inner1 });
            var inner4 = new NodeLayer("Inner 4", 10, new[] { inner3 });
            var output = new NodeLayer("Output", 10, new[] { inner2, inner4 });

            var backPropGroupData = BackpropagationMethods.GenerateBackpropagationBindingModel(output);

            Assert.AreEqual(inputGroup, backPropGroupData.NodeGroup);
            CollectionAssert.Contains(backPropGroupData.FeedingGroups.Select(x => x.NodeGroup).ToArray(), inner1);
            CollectionAssert.Contains(backPropGroupData.FeedingGroups.Select(x => x.NodeGroup).ToArray(), inner2);
            Assert.AreEqual(inner3, backPropGroupData.FeedingGroups.First(fg => fg.NodeGroup.Name == "Inner 1").FeedingGroups[0].NodeGroup);
            Assert.AreEqual(inner4, backPropGroupData.FeedingGroups.First(fg => fg.NodeGroup.Name == "Inner 1").FeedingGroups[0].FeedingGroups[0].NodeGroup);
            Assert.AreEqual(output, backPropGroupData.FeedingGroups.First(fg => fg.NodeGroup.Name == "Inner 2").FeedingGroups[0].NodeGroup);
            Assert.AreEqual(output, backPropGroupData.FeedingGroups.First(fg => fg.NodeGroup.Name == "Inner 1").FeedingGroups[0].FeedingGroups[0].FeedingGroups[0].NodeGroup);
        }
    }
}
