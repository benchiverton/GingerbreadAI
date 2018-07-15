using NeuralNetwork.Data;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NeuralNetwork.Library
{
    public static class NodeGroupMethods
    {
        public static NodeGroup[] GetAllGroupsInSystem(NodeGroup nodeGroup)
        {
            var result = new List<NodeGroup> { nodeGroup };

            foreach (var prevNodeGroup in nodeGroup.PreviousGroups)
            {
                if (result.All(ng => ng.Name != prevNodeGroup.Name))
                {
                    result.Add(prevNodeGroup);
                }
                var nodeBeforeGroups = GetAllGroupsInSystem(prevNodeGroup);
                foreach (var nodeBeforeGroup in nodeBeforeGroups)
                {
                    if (result.All(ng => ng.Name != nodeBeforeGroup.Name))
                    {
                        result.Add(nodeBeforeGroup);
                    }
                }
            }
            return result.ToArray();
        }
    }
}
