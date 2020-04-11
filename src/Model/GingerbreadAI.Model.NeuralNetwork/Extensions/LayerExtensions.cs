using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.NeuralNetwork.Extensions
{
    public static class LayerExtensions
    {
        /// <summary>
        /// Initialises each Node in the layer with random weights.
        /// </summary>
        public static void Initialise(this Layer layer, Random rand)
        {
            foreach (var node in layer.Nodes)
            {
                node.Initialise(rand, layer.InitialisationFunction, layer.Nodes.Length);
            }
            foreach (var nodeGroupPrev in layer.PreviousLayers)
            {
                nodeGroupPrev.Initialise(rand);
            }
        }

        public static double[] GetResults(this Layer layer, double[] inputs)
        {
            layer.CalculateOutputs(inputs);
            return layer.Nodes.Select(n => n.Output).ToArray();
        }

        public static double[] GetResults(this Layer layer, Dictionary<Layer, double[]> inputs)
        {
            layer.CalculateOutputs(inputs);
            return layer.Nodes.Select(n => n.Output).ToArray();
        }

        public static double GetResult(this Layer layer, int inputIndex, int outputIndex, double inputValue = 1)
        {
            layer.CalculateIndexedOutput(inputIndex, outputIndex, inputValue);
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

        // Use this when multi-threading the same network
        public static Layer CloneWithSameWeightValueReferences(this Layer layer)
        {
            return RecurseCloneWithSameWeightValueReferences(layer);
        }

        public static void SaveNetwork(this Layer layer, string location)
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

        public static Layer LoadNetwork(string location)
        {
            using (var byteStream = File.OpenRead(location))
            {
                using (var memoryStream = new MemoryStream())
                {
                    byteStream.CopyTo(memoryStream);
                    memoryStream.Position = 0;

                    var formatter = new BinaryFormatter();
                    return (Layer)formatter.Deserialize(memoryStream);
                }
            }
        }

        #region Private Methods


        private static void Initialise(this Node node, Random rand, Func<Random, int, int, double>  initialisationFunction, int nodeCount)
        {
            if (node == null) return;
            var feedingNodes = node.Weights.Count;
            foreach (var prevNode in node.Weights.Keys.ToList())
            {
                node.Weights[prevNode].Adjust(initialisationFunction.Invoke(rand, feedingNodes, nodeCount));
            }
            var biasWeightKeys = new List<Layer>(node.BiasWeights.Keys.ToList());
            foreach (var biasWeightKey in biasWeightKeys)
            {
                node.BiasWeights[biasWeightKey].Adjust(initialisationFunction.Invoke(rand, feedingNodes, nodeCount));
            }
        }

        private static Layer RecurseCloneWithSameWeightValueReferences(Layer layer)
        {
            if (!layer.PreviousLayers.Any())
            {
                var newInputLayer = new Layer()
                {
                    Nodes = new Node[layer.Nodes.Length],
                    PreviousLayers = new Layer[0],
                    ActivationFunctionType = layer.ActivationFunctionType,
                    InitialisationFunctionType = layer.InitialisationFunctionType
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

            var newLayer = new Layer
            {
                PreviousLayers = clonedPreviousLayers.ToArray(),
                Nodes = new Node[layer.Nodes.Length],
                ActivationFunctionType = layer.ActivationFunctionType,
                InitialisationFunctionType = layer.InitialisationFunctionType
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
                        if (layer.Nodes[i].Weights.TryGetValue(layer.PreviousLayers[j].Nodes[k], out var weight))
                        {
                            newNode.Weights.Add(newLayer.PreviousLayers[j].Nodes[k], weight);
                        }
                    }
                    if (layer.Nodes[i].BiasWeights.TryGetValue(layer.PreviousLayers[j], out var biasWeight))
                    {
                        newNode.BiasWeights.Add(newLayer.PreviousLayers[j], biasWeight);
                    }
                }

                newLayer.Nodes[i] = newNode;
            }

            return newLayer;
        }

        #endregion
    }
}
