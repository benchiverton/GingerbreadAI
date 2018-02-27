using System;
using NeuralNetwork.Data;
using NeuralNetwork.Exceptions;

namespace NeuralNetwork.Library
{
    public class NodeNetworkCalculations
    {
        /// <summary>
        ///     Adds a NodeGroup to the network.
        /// </summary>
        /// <param name="nodeGroup"></param>
        /// <param name="nodeNetwork"></param>
        public static void AddNodeGroup(NodeGroup nodeGroup, NodeNetwork nodeNetwork)
        {
            if (nodeNetwork.Groups == null)
            {
                nodeNetwork.Groups = new NodeGroup[1];
                nodeNetwork.Groups[0] = nodeGroup;
            }
            else
            {
                Array.Resize(ref nodeNetwork.Groups, nodeNetwork.Groups.Length + 1);
                nodeNetwork.Groups[nodeNetwork.Groups.Length - 1] = nodeGroup;
            }
        }

        public static double[] GetResult(double[] inputs, NodeNetwork nodeNetwork)
        {
            if (inputs.Length != nodeNetwork.Groups[0].Nodes.Length)
                throw new NodeNetworkException("Please enter the correct amount of inputs for your network.");

            return NodeGroupCalculations.GetResult(inputs, nodeNetwork.Groups[nodeNetwork.Groups.Length - 1]);
        }
    }
}
