using Backpropogation.Library;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Data;
using System.Linq;

namespace BackpropogationTest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void GenerateBackpropogationGroupsDataTest()
        {
            var inputGroup = new NodeGroup("Input Group", 1);
            var inner1 = new NodeGroup("Inner 1", 10, new[] { inputGroup });
            var inner2 = new NodeGroup("Inner 2", 10, new[] { inputGroup });
            var inner3 = new NodeGroup("Inner 3", 10, new[] { inner1 });
            var inner4 = new NodeGroup("Inner 4", 10, new[] { inner3 });
            var output = new NodeGroup("Output", 10, new[] { inner2, inner4 });

            var backPropGroupData = BackpropogationMethods.GenerateBackpropogationGroupsData(output);

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
