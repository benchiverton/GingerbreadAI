using System.Text;

namespace NeuralNetwork.Data
{
    /// <summary>
    ///     The NodeGroups are stored in a Linked List, so it needs to contain a reference to the NodeGroup before it.
    /// </summary>
    public class NodeGroup
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
        public NodeGroup[] PreviousGroups { get; set; }

        /// <summary>
        ///     Should be used to set up the input
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        public NodeGroup(string name, int nodeCount)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            PreviousGroups = null;
        }

        /// <summary>
        ///     Constructs a NodeGroup, initialising each node with the correct amount of Weights.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        /// <param name="previousGroups"></param>
        public NodeGroup(string name, int nodeCount, NodeGroup[] previousGroups)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            for (var i = 0; i < nodeCount; i++)
                Nodes[i] = new Node(previousGroups);
            PreviousGroups = previousGroups;
        }

        /// <summary>
        ///     Initialises this NodeGroup with the parameters supplied.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodes"></param>
        /// <param name="previousGroup"></param>
        public NodeGroup(string name, Node[] nodes, NodeGroup[] previousGroup)
        {
            Name = name;
            Nodes = nodes;
            PreviousGroups = previousGroup;
        }

        public override string ToString()
        {
            var s = new StringBuilder($"Node Group: {Name}\n");
            for (var i = 0; i < Nodes.Length; i++)
                s.Append($"Node {i}:\n{Nodes[i]}");
            if (PreviousGroups != null)
            {
                s.Append("Previous Groups:\n");
                foreach (var nodeGroup in PreviousGroups)
                    s.Append($"{nodeGroup.Name}\n");
            }
            s.Append("----------\n");
            return s.ToString();
        }
    }
}