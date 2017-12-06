using System;
using System.Text;

namespace NeuralNetwork.Nodes
{
    // Stored as a doubly linked list, so that a layer knows where it is in relation to the model
    public class NodeLayer
    {
        /// <summary>
        /// The name of the node (purely for helping you design and navagate your network).
        /// </summary>
        public string Name;
        /// <summary>
        /// An array of the nodes within this layer.
        /// </summary>
        public Node[] Nodes;
        /// <summary>
        /// An array containing the NodeLayers that feed into this one.
        /// </summary>
        public NodeLayer[] PreviousLayers;

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
        }

        /// <summary>
        /// Constructs a NodeLayer, initialising each node with the correct amount of Weights.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        /// <param name="previousLayers"></param>
        public NodeLayer(string name, int nodeCount, NodeLayer[] previousLayers)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            for (var i = 0; i < nodeCount; i++)
            {
                Nodes[i] = new Node(previousLayers);
            }
            PreviousLayers = previousLayers;
        }

        public void Initialise(Random rand)
        {
            foreach (var node in Nodes)
            {
                node?.Initialise(rand);
            }
        }

        public override string ToString()
        {
            var s = new StringBuilder($"Node Layer: {Name}\n");
            for(var i=0; i<Nodes.Length; i++)
            {
                s.Append($"Node {i}:\n{Nodes[i]}");
            }
            if (PreviousLayers != null)
            {
                s.Append("Previous Layers:\n");
                foreach (var nodeLayer in PreviousLayers)
                {
                    s.Append($"{nodeLayer.Name}\n");
                }
            }
            s.Append("----------\n");
            return s.ToString();
        }
    }
}
