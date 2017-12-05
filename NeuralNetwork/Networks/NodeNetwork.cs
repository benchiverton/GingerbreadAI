using NeuralNetwork.Exceptions;
using NeuralNetwork.Nodes;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Networks
{
    class NodeNetwork
    {
        public List<NodeLayer> Layers;

        public NodeNetwork(List<NodeLayer> layers)
        {
            Layers = layers;
        }

        public void AddInput(NodeLayer input)
        {
            Layers.Add(input);
        }

        public void AddLayer(NodeLayer layer, string[] feedsFrom)
        {
            foreach (NodeLayer l in Layers)
            {
                if(l.Name == layer.Name)
                {
                    throw new NodeNetworkException($"A layer with the name '{layer.Name}' already exists.");
                }
            }

            //TODO: Add this layer to NextLayers, and initialize PreviousLayers.
        }

        // method to order the list so getting result is rly quick (maybe so backprop will be faster also?)
        // - is this possible mathematically?

        // method to get the result
    }
}
