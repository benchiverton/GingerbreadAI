using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using Backpropogation.Data;
using Backpropogation.Exceptions;
using NeuralNetwork.Data;
using NeuralNetwork.Library;

namespace Backpropogation.Library
{
    public class Backpropogation
    {
        public double LearningRate { get; set; }

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
            throw new NotImplementedException();
        }
    }
}