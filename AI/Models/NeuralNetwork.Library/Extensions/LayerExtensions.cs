using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using NeuralNetwork.Data;

namespace NeuralNetwork.Library.Extensions
{
    public static class LayerExtensions
    {
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

        public static Layer GetCopyWithReferences(this Layer layer)
        {
            return RecurseSettingWeightsToZero(layer);
        }

        private static Layer RecurseSettingWeightsToZero(Layer layer)
        {
            var newLayer = new Layer
            {
                Nodes = new Node[layer.Nodes.Length],
                PreviousLayers = new Layer[layer.PreviousLayers.Length]
            };

            for (var i = 0; i < layer.Nodes.Length; i++)
            {
                var newNode = new Node
                {
                    Weights = new Dictionary<Node, double>(),
                    BiasWeights = new Dictionary<Layer, double>(),
                    Output = 0
                };

                foreach (var weightKey in layer.Nodes[i].Weights.Keys)
                {
                    newNode.Weights.Add(weightKey, 0);
                }

                foreach (var biasWeightKey in layer.Nodes[i].BiasWeights.Keys)
                {
                    newNode.BiasWeights.Add(biasWeightKey, 0);
                }

                newLayer.Nodes[i] = newNode;
            }

            for (var i = 0; i < layer.PreviousLayers.Length; i++)
            {
                newLayer.PreviousLayers[i] = RecurseSettingWeightsToZero(layer.PreviousLayers[i]);
            }

            return newLayer;
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
    }
}
