using NeuralNetwork.Data;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Library
{
    public static class NodeLayerMethods
    {
        public static NodeLayer[] GetAllGroupsInSystem(NodeLayer nodeGroup)
        {
            var result = new List<NodeLayer> { nodeGroup };

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
