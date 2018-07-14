using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Data;

namespace Backpropogation.Data
{
    public class BackpropogationGroupData
    {
        // we need a reference to the feeding groups in order to peform the backwards pass.
        public NodeGroup NodeGroup { get; set; }
        public BackpropogationGroupData[] FeedingGroups { get; set; }

        public BackpropogationGroupData(NodeGroup nodeGroup)
        {
            NodeGroup = nodeGroup;
            FeedingGroups = new BackpropogationGroupData[0];
        }
    }
}
