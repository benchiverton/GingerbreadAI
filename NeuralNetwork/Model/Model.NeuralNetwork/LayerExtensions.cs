using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using Library.Computations;
using Model.NeuralNetwork.Models;

namespace Model.NeuralNetwork
{
    public static class LayerExtensions
    {
        /// <summary>
        ///     Initialises each Node in the layer with random weights.
        /// </summary>
        public static void Initialise(this Layer layer, Random rand)
        {
            foreach (var node in layer.Nodes)
            {
                node.Initialise(rand);
            }
            foreach (var nodeGroupPrev in layer.PreviousLayers)
            {
                nodeGroupPrev.Initialise(rand);
            }
        }

        public static double[] GetResults(this Layer layer, double[] inputs)
        {
            layer.PopulateAllOutputs(inputs);
            return layer.Nodes.Select(n => n.Output).ToArray();
        }

        public static double[] GetResults(this Layer layer, Dictionary<Layer, double[]> inputs)
        {
            layer.PopulateAllOutputs(inputs);
            return layer.Nodes.Select(n => n.Output).ToArray();
        }

        public static double GetResult(this Layer layer, int inputIndex, int outputIndex, double inputValue = 1)
        {
            layer.PopulateIndexedOutput(inputIndex, outputIndex, inputValue);
            return layer.Nodes[outputIndex].Output;
        }

        public static Layer DeepCopy(this Layer layer)
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);
                ms.Position = 0;
                return (Layer)formatter.Deserialize(ms);
            }
        }

        public static Layer CloneWithSameWeightKeyReferences(this Layer layer)
        {
            return RecurseCloneWithSameWeightKeyReferences(layer);
        }

        // Use this when multi-threading the same network
        public static Layer CloneWithSameWeightValueReferences(this Layer layer)
        {
            return RecurseCloneWithSameWeightValueReferences(layer);
        }

        public static void Save(this Layer layer, string location)
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);
                using (var fileStream = File.Create(location))
                {
                    ms.WriteTo(fileStream);
                }
            }
        }

        #region Private Methods


        /// <summary>
        ///     Initialises a Node with random weights (using He-et-al Initialization).
        /// </summary>
        private static void Initialise(this Node node, Random rand)
        {
            if (node == null) return;
            var feedingNodes = node.Weights.Count;
            foreach (var prevNode in node.Weights.Keys.ToList())
            {
                node.Weights[prevNode].Value = NetworkCalculations.GetWeightedInitialisation(rand, feedingNodes);
            }
            var biasWeightKeys = new List<Layer>(node.BiasWeights.Keys.ToList());
            foreach (var biasWeightKey in biasWeightKeys)
            {
                node.BiasWeights[biasWeightKey].Value = NetworkCalculations.GetWeightedInitialisation(rand, feedingNodes);
            }
        }

        private static Layer RecurseCloneWithSameWeightKeyReferences(Layer layer)
        {
            var newLayer = new Layer
            {
                Name = $"{layer.Name}_CLONE",
                Nodes = new Node[layer.Nodes.Length],
                PreviousLayers = new Layer[layer.PreviousLayers.Length]
            };

            for (var i = 0; i < layer.Nodes.Length; i++)
            {
                var newNode = new Node
                {
                    Weights = new Dictionary<Node, Weight>(),
                    BiasWeights = new Dictionary<Layer, Weight>()
                };

                foreach (var weightKey in layer.Nodes[i].Weights.Keys)
                {
                    newNode.Weights.Add(weightKey, new Weight(0));
                }

                foreach (var biasWeightKey in layer.Nodes[i].BiasWeights.Keys)
                {
                    newNode.BiasWeights.Add(biasWeightKey, new Weight(0));
                }

                newLayer.Nodes[i] = newNode;
            }

            for (var i = 0; i < layer.PreviousLayers.Length; i++)
            {
                newLayer.PreviousLayers[i] = RecurseCloneWithSameWeightKeyReferences(layer.PreviousLayers[i]);
            }

            return newLayer;
        }

        private static Layer RecurseCloneWithSameWeightValueReferences(Layer layer)
        {
            if (!layer.PreviousLayers.Any())
            {
                var newInputLayer = new Layer()
                {
                    Name = $"{layer.Name}_CLONE",
                    Nodes = new Node[layer.Nodes.Length],
                    PreviousLayers = new Layer[0]
                };

                for (var i = 0; i < layer.Nodes.Length; i++)
                {
                    newInputLayer.Nodes[i] = new Node
                    {
                        Weights = new Dictionary<Node, Weight>(),
                        BiasWeights = new Dictionary<Layer, Weight>()
                    };
                }

                return newInputLayer;
            }

            var clonedPreviousLayers = new List<Layer>();
            foreach (var previousLayer in layer.PreviousLayers)
            {
                clonedPreviousLayers.Add(RecurseCloneWithSameWeightValueReferences(previousLayer));
            }

            var newLayer = new Layer()
            {
                Name = $"{layer.Name}_CLONE",
                PreviousLayers = clonedPreviousLayers.ToArray(),
                Nodes = new Node[layer.Nodes.Length]
            };

            for (var i = 0; i < layer.Nodes.Length; i++)
            {
                var newNode = new Node()
                {
                    Weights = new Dictionary<Node, Weight>(),
                    BiasWeights = new Dictionary<Layer, Weight>()
                };

                for (var j = 0; j < layer.PreviousLayers.Length; j++)
                {
                    for (var k = 0; k < layer.PreviousLayers[j].Nodes.Length; k++)
                    {
                        newNode.Weights.Add(newLayer.PreviousLayers[j].Nodes[k], layer.Nodes[i].Weights[layer.PreviousLayers[j].Nodes[k]]);
                    }

                    newNode.BiasWeights.Add(newLayer.PreviousLayers[j], layer.Nodes[i].BiasWeights[layer.PreviousLayers[j]]);
                }

                newLayer.Nodes[i] = newNode;
            }

            return newLayer;
        }

        #endregion
    }
}
