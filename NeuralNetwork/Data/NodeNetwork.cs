using System;
using System.Text;
using NeuralNetwork.Exceptions;

namespace NeuralNetwork.Data
{
    public class NodeNetwork
    {
        /// <summary>
        ///     An array containing all of the NodeLayers within the network. I feel like the input might not be needed...
        /// </summary>
        public NodeLayer[] Layers;

        public override string ToString()
        {
            var s = new StringBuilder("Your Network:\n");
            foreach (var nodeLayer in Layers)
                s.Append($"{nodeLayer}");
            return s.ToString();
        }
    }
}