using System;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using DeepLearning.Data;
using DeepLearning.Exceptions;
using NeuralNetwork.Data;
using NeuralNetwork.Library;

namespace DeepLearning
{
    public class Backpropogation
    {
        public double LearningRate { get; set; }

        public BackpropogationGroupData[] GetBackpropogationGroupsData(NodeGroup nodeGroup)
        {
            throw new NotImplementedException();
        }

        public void TeachNetwork(BackpropogationGroupData[] network, double[] inputs, double[] targetOutputs)
        {
            if (network.First().NodeGroup.Nodes.Length != inputs.Length)
                throw new IncorrectArrayLengthException(
                    "Your inputs vector was not of the correct length for backpropogation.");
            if (network.Last().NodeGroup.Nodes.Length != targetOutputs.Length)
                throw new IncorrectArrayLengthException(
                    "Your trueOutputs vector was not of the correct length for backpropogation.");

            Backpropogate(network.First());
        }

        // needs to return the sumDeltaWeight
        public static void Backpropogate(BackpropogationGroupData workingGroup)
        {
            if (workingGroup.FeedingGroups == null)
            {
                //workingGroup.SumDeltaWeight = 0;
                return;
            }

            foreach (var feedingGroup in workingGroup.FeedingGroups)
            {
                //if (feedingGroup.SumDeltaWeight == null)
                //{
                    
                //}
            }
        }
    }
}