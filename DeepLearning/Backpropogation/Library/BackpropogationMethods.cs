using Backpropogation.Data;
using NeuralNetwork.Data;
using NeuralNetwork.Library;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Backpropogation.Library
{
    public static class BackpropogationMethods
    {
        public static BackpropogationGroupData GenerateBackpropogationGroupsData(NodeGroup nodeGroup)
        {
            var allNodeGroups = NodeGroupMethods.GetAllGroupsInSystem(nodeGroup);
            var inputNodeGroup = allNodeGroups.First(ng => ng.PreviousGroups.Length == 0);
            var inputBackpropData = new BackpropogationGroupData(inputNodeGroup);

            BuildBackpropogationGroupsData(allNodeGroups, inputBackpropData);

            return inputBackpropData;
        }

        public static void BuildBackpropogationGroupsData(NodeGroup[] nodeGroups, BackpropogationGroupData inputNodeGroup)
        {
            foreach (var nodeGroup in nodeGroups.Where(ng => ng.PreviousGroups.Contains(inputNodeGroup.NodeGroup)))
            {
                var feedingGroups = inputNodeGroup.FeedingGroups;
                Array.Resize(ref feedingGroups, inputNodeGroup.FeedingGroups.Length + 1);
                inputNodeGroup.FeedingGroups = feedingGroups;

                var x = new BackpropogationGroupData(nodeGroup);
                BuildBackpropogationGroupsData(nodeGroups, x);
                inputNodeGroup.FeedingGroups[inputNodeGroup.FeedingGroups.Length - 1] = x;
            }
        }
    }
}
