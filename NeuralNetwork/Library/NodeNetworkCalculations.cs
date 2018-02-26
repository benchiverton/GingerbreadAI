using System;
using NeuralNetwork.Data;
using NeuralNetwork.Exceptions;

namespace NeuralNetwork.Library
{
    public class NodeNetworkCalculations
    {
        /// <summary>
        ///     Adds a NodeLayer to the network.
        /// </summary>
        /// <param name="nodeLayer"></param>
        /// <param name="nodeNetwork"></param>
        public static void AddNodeLayer(NodeLayer nodeLayer, NodeNetwork nodeNetwork)
        {
            if (nodeNetwork.Layers == null)
            {
                nodeNetwork.Layers = new NodeLayer[1];
                nodeNetwork.Layers[0] = nodeLayer;
            }
            else
            {
                Array.Resize(ref nodeNetwork.Layers, nodeNetwork.Layers.Length + 1);
                nodeNetwork.Layers[nodeNetwork.Layers.Length - 1] = nodeLayer;
            }
        }

        public static double[] GetResult(double[] inputs, NodeNetwork nodeNetwork)
        {
            if (inputs.Length != nodeNetwork.Layers[0].Nodes.Length)
                throw new NodeNetworkException("Please enter the correct amount of inputs for your network.");

            return NodeLayerCalculations.GetResult(inputs, nodeNetwork.Layers[nodeNetwork.Layers.Length - 1]);
        }
    }
}
