using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes
{
    // Stored as a doubly linked list, so that a layer knows where it is in relation to the model
    class NodeLayer
    {
        /// <summary>
        /// The name of the node (purely for helping you design and navagate your network).
        /// </summary>
        public string Name;
        /// <summary>
        /// An array of the nodes within this layer.
        /// </summary>
        public Node[] Nodes;
        // needed for backpropogation?
        /// <summary>
        /// An array containing the NodeLayers that feed into this one.
        /// </summary>
        public NodeLayer[] PreviousLayers;
        // needed for getting the result?
        /// <summary>
        /// An array containing the NodeLayers which this one feed into.
        /// </summary>
        public NodeLayer[] NextLayers;

        /// <summary>
        /// Should be used to set up the input
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        public NodeLayer(string name, int nodeCount)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            PreviousLayers = null;
            NextLayers = null;
        }

        public void AddPreviousLayer(NodeLayer layer)
        {
            if (NextLayers == null)
            {
                PreviousLayers = new NodeLayer[1];
            }
            else
            {
                Array.Resize(ref PreviousLayers, PreviousLayers.Length + 1);
            }
            PreviousLayers[PreviousLayers.Length - 1] = layer;
        }

        public void AddNextLayer(NodeLayer layer)
        {
            if (NextLayers == null)
            {
                NextLayers = new NodeLayer[1];
            }
            else
            {
                Array.Resize(ref NextLayers, NextLayers.Length + 1);
            }
            NextLayers[NextLayers.Length - 1] = layer;
        }
    }
}
