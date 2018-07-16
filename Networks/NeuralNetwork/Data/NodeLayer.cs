using System.Text;

namespace NeuralNetwork.Data
{
    /// <summary>
    ///     The NodeGroups are stored in a Linked List, so it needs to contain a reference to the NodeGroup before it.
    /// </summary>
    public class NodeLayer
    {
        /// <summary>
        ///     The name of the node (purely for helping you design and navagate your network).
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        ///     An array of the nodes within this group.
        /// </summary>
        public Node[] Nodes { get; set; }

        /// <summary>
        ///     An array containing the NodeGroups that feed into this one.
        /// </summary>
        public NodeLayer[] PreviousGroups { get; set; }

        /// <summary>
        ///     The current output of this layer
        /// </summary>
        public double[] Outputs { get; set; }

        /// <summary>
        ///     Should be used to set up the input
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        public NodeLayer(string name, int nodeCount)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            PreviousGroups = new NodeLayer[0];
            Outputs = new double[nodeCount];
        }

        /// <summary>
        ///     Constructs a NodeGroup, initialising each node with the correct amount of Weights.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        /// <param name="previousGroups"></param>
        public NodeLayer(string name, int nodeCount, NodeLayer[] previousGroups)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            for (var i = 0; i < nodeCount; i++)
            {
                Nodes[i] = new Node(previousGroups);
            }
            PreviousGroups = previousGroups;
            Outputs = new double[nodeCount];
        }

        /// <summary>
        ///     Initialises this NodeGroup with the parameters supplied.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodes"></param>
        /// <param name="previousGroup"></param>
        public NodeLayer(string name, Node[] nodes, NodeLayer[] previousGroup)
        {
            Name = name;
            Nodes = nodes;
            PreviousGroups = previousGroup;
            Outputs = new double[nodes.Length];
        }

        public string ToString(bool recurse = false, int layer = 0)
        {
            string indentation = "";
            for (int i = 0; i < layer; i++)
            {
                indentation += "    ";
            }

            var s = new StringBuilder($"{indentation}Node Group: {Name}; Node count: {Nodes.Length}\n");
            s.Append($"{indentation}Previous Groups:\n");

            layer++;
            foreach (var nodeGroup in PreviousGroups)
            {
                if (recurse)
                {
                    s.Append(nodeGroup.ToString(true, layer));
                }
                else
                {
                    s.Append($"{indentation}Node Group: {nodeGroup.Name}; Node count: {nodeGroup.Nodes.Length}\n");
                }
            }
            return s.ToString();
        }
    }
}