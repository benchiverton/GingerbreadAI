using System.Linq;
using DeepLearning.Backpropagation.Interfaces;
using DeepLearning.Backpropagation.Models;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;

namespace DeepLearning.Backpropagation.Extensions
{
    public static class LayerExtensions
    {
        public static void AddMomentumRecursively(this Layer layer)
        {
            layer.AddMomentum();
            foreach (var previousLayer in layer.PreviousLayers)
            {
                previousLayer.AddMomentumRecursively();
            }
        }

        public static void AddMomentum(this Layer layer)
        {
            foreach (var node in layer.Nodes)
            {
                var prevNodes = node.Weights.Keys.ToArray();
                foreach (var prevNode in prevNodes)
                {
                    node.Weights[prevNode] = GetWeightWithMomentum(node.Weights[prevNode]);
                }
                var prevLayers = node.BiasWeights.Keys.ToArray();
                foreach (var prevLayer in prevLayers)
                {
                    node.BiasWeights[prevLayer] = GetWeightWithMomentum(node.BiasWeights[prevLayer]);
                }
            }
        }

        private static Weight GetWeightWithMomentum(Weight weightWithoutMomentum)
        {
            return weightWithoutMomentum switch
            {
                WeightWithPooling weightWithPooling => new WeightWithPoolingAndMomentum(weightWithPooling),
                _ => new WeightWithMomentum(weightWithoutMomentum.Value),
            };
        }
    }
}
