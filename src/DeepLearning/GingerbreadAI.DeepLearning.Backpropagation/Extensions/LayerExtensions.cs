using System.Linq;
using GingerbreadAI.DeepLearning.Backpropagation.Models;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.DeepLearning.Backpropagation.Extensions;

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
                node.Weights[prevNode] = new WeightWithMomentum(node.Weights[prevNode].Value);
            }
            var prevLayers = node.BiasWeights.Keys.ToArray();
            foreach (var prevLayer in prevLayers)
            {
                node.BiasWeights[prevLayer] = new WeightWithMomentum(node.BiasWeights[prevLayer].Value);
            }
        }
    }
}
