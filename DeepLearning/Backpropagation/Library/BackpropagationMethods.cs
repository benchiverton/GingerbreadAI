using Backpropagation.Data;
using NeuralNetwork.Data;
using NeuralNetwork.Library;
using System;
using System.Linq;

namespace Backpropagation.Library
{
    public static class BackpropagationMethods
    {
        public static BackpropagationBindingModel GenerateBackpropagationBindingModel(NodeLayer nodeGroup)
        {
            var allNodeGroups = NodeLayerMethods.GetAllGroupsInSystem(nodeGroup);
            var inputNodeGroup = allNodeGroups.First(ng => ng.PreviousGroups.Length == 0);
            var inputBackpropData = new BackpropagationBindingModel(inputNodeGroup);

            BuildBackpropagationBindingModel(allNodeGroups, inputBackpropData);

            return inputBackpropData;
        }

        private static void BuildBackpropagationBindingModel(NodeLayer[] nodeGroups, BackpropagationBindingModel inputNodeGroup)
        {
            foreach (var nodeGroup in nodeGroups.Where(ng => ng.PreviousGroups.Contains(inputNodeGroup.BoundNodeLayer)))
            {
                var feedingGroups = inputNodeGroup.FeedingGroups;
                Array.Resize(ref feedingGroups, inputNodeGroup.FeedingGroups.Length + 1);
                inputNodeGroup.FeedingGroups = feedingGroups;

                var x = new BackpropagationBindingModel(nodeGroup);
                BuildBackpropagationBindingModel(nodeGroups, x);
                inputNodeGroup.FeedingGroups[inputNodeGroup.FeedingGroups.Length - 1] = x;
            }
        }

        //public static NodeLayer GetOutputNodeGroup(BackpropagationBindingModel backpropGroupData)
        //{
        //    while (backpropGroupData.FeedingGroups.Length != 0)
        //    {
        //        backpropGroupData = backpropGroupData.FeedingGroups[0];
        //    }

        //    return backpropGroupData.NodeGroup;
        //}
    }
}
