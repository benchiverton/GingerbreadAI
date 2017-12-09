using System;
using System.Text;
using NeuralNetwork.Exceptions;
using NeuralNetwork.Nodes;

namespace NeuralNetwork.Networks
{
    public class NodeNetwork
    {
        /// <summary>
        ///     An array containing all of the NodeLayers within the network. I feel like the input might not be needed...
        /// </summary>
        public NodeLayer[] Layers;

        /// <summary>
        ///     Base constructor.
        /// </summary>
        public NodeNetwork()
        {
            Layers = null;
        }

        /// <summary>
        ///     Initialises the layers in this network to the ones supplied.
        /// </summary>
        /// <param name="layers"></param>
        public NodeNetwork(NodeLayer[] layers)
        {
            Layers = layers;
        }

        /// <summary>
        ///     Adds a NodeLayer to the network.
        /// </summary>
        /// <param name="nodeLayer"></param>
        public void AddNodeLayer(NodeLayer nodeLayer)
        {
            if (Layers == null)
            {
                Layers = new NodeLayer[1];
                Layers[0] = nodeLayer;
            }
            else
            {
                Array.Resize(ref Layers, Layers.Length + 1);
                Layers[Layers.Length - 1] = nodeLayer;
            }
        }

        /// <summary>
        ///     Initialises each node within the network with random weights.
        /// </summary>
        /// <param name="rand"></param>
        public void Initialise(Random rand)
        {
            foreach (var layer in Layers)
                layer.Initialise(rand);
        }

        public double[] GetResult(double[] inputs)
        {
            if (inputs.Length != Layers[0].Nodes.Length)
                throw new NodeNetworkException("Please enter the correct amount of inputs for your network.");

            return Layers[Layers.Length - 1].GetResult(inputs);
        }

        public override string ToString()
        {
            var s = new StringBuilder("Your Network:\n");
            foreach (var nodeLayer in Layers)
                s.Append($"{nodeLayer}");
            return s.ToString();
        }
    }
}