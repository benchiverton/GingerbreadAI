using System.Text;

namespace NeuralNetwork.Data
{
    /// <summary>
    ///     The NodeLayers are stored in a Linked List, so it needs to contain a reference to the NodeLayer before it.
    /// </summary>
    public class NodeLayer
    {
        /// <summary>
        ///     The name of the node (purely for helping you design and navagate your network).
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        ///     An array of the nodes within this layer.
        /// </summary>
        public Node[] Nodes { get; set; }

        /// <summary>
        ///     An array containing the NodeLayers that feed into this one.
        /// </summary>
        public NodeLayer[] PreviousLayers { get; set; }

        /// <summary>
        ///     Should be used to set up the input
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
        ///     Constructs a NodeLayer, initialising each node with the correct amount of Weights.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        /// <param name="previousLayers"></param>
        public NodeLayer(string name, int nodeCount, NodeLayer[] previousLayers)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            for (var i = 0; i < nodeCount; i++)
                Nodes[i] = new Node(previousLayers);
            PreviousLayers = previousLayers;
        }

        /// <summary>
        ///     Initialises this NodeLayer with the parameters supplied.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodes"></param>
        /// <param name="previousLayer"></param>
        public NodeLayer(string name, Node[] nodes, NodeLayer[] previousLayer)
        {
            Name = name;
            Nodes = nodes;
            PreviousLayers = previousLayer;
        }

        public override string ToString()
        {
            var s = new StringBuilder($"Node Layer: {Name}\n");
            for (var i = 0; i < Nodes.Length; i++)
                s.Append($"Node {i}:\n{Nodes[i]}");
            if (PreviousLayers != null)
            {
                s.Append("Previous Layers:\n");
                foreach (var nodeLayer in PreviousLayers)
                    s.Append($"{nodeLayer.Name}\n");
            }
            s.Append("----------\n");
            return s.ToString();
        }
    }
}