using System;
using NeuralNetwork.Nodes;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Networks
{
    public class NodeNetwork
    {
        public List<NodeLayer> Layers;


        /// <summary>
        /// Base constructor.
        /// </summary>
        public NodeNetwork()
        {
            Layers = new List<NodeLayer>();
        }

        /// <summary>
        /// Initialises the layers in this network to the ones supplied.
        /// </summary>
        /// <param name="layers"></param>
        public NodeNetwork(List<NodeLayer> layers)
        {
            Layers = layers;
        }

        /// <summary>
        /// Adds a NodeLayer to the network.
        /// </summary>
        /// <param name="nodeLayer"></param>
        public void AddNodeLayer(NodeLayer nodeLayer)
        {
            Layers.Add(nodeLayer);
        }

        // method to initialize the weights (ie - set them to random)
        public void Initialise(Random rand)
        {
            foreach (var layer in Layers)
            {
                layer.Initialise(rand);
            }
        }

        // method to get the result

        public override string ToString()
        {
            var s = new StringBuilder("Your Network:\n");
            foreach (var nodeLayer in Layers)
            {
                s.Append($"{nodeLayer}");
            }
            return s.ToString();
        }
    }
}
