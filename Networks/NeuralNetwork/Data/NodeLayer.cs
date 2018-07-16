using System.Text;

namespace NeuralNetwork.Data
{
    using System.Collections.Generic;

    public class NodeLayer
    {
        /// <summary>
        ///     The name of the node.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        ///     An array of the nodes within this group.
        /// </summary>
        public Node[] Nodes { get; set; }

        /// <summary>
        ///     An array containing the NodeGroups that feed into this one.
        /// </summary>
        public NodeLayer[] PreviousLayers { get; set; }

        public NodeLayer()
        {
            // default constructor
        }

        public NodeLayer(string name, int nodeCount, NodeLayer[] previousGroups)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            for (var i = 0; i < nodeCount; i++)
            {
                Nodes[i] = new Node(previousGroups);
            }
            PreviousLayers = previousGroups;
            foreach (var node in Nodes)
            {
                node.Output = 0;
            }
        }

        public string ToString(bool recurse = false, int layer = 0)
        {
            var indentation = "";
            for (var i = 0; i < layer; i++)
            {
                indentation += "    ";
            }

            var s = new StringBuilder($"{indentation}Node Group: {Name}; Node count: {Nodes.Length}\n");
            s.Append($"{indentation}Previous Groups:\n");

            layer++;
            foreach (var nodeGroup in PreviousLayers)
            {
                s.Append(recurse
                    ? nodeGroup.ToString(true, layer)
                    : $"{indentation}Node Group: {nodeGroup.Name}; Node count: {nodeGroup.Nodes.Length}\n");
            }
            return s.ToString();
        }
    }
}