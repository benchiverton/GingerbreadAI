using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Data;

namespace DeepLearning.Data
{
    public class BackpropogationGroupData
    {
        public NodeGroup NodeGroup { get; set; }
        public BackpropogationGroupData[] FeedingGroups { get; set; }
        public double[] NodeOutputs { get; set; }
    }
}
