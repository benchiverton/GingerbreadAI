using System;
using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;

public class PooledNode : Node
{
    public PooledNode(List<Node> underlyingNodes) => UnderlyingNodes = underlyingNodes;

    public List<Node> UnderlyingNodes { get; init; }

    public override void CalculateOutput(Func<double, double> activationFunction)
    {
        // Calculates output for underlying nodes and sets output/weights/biasweights to the max node.
        // This means backpropagation etc will only update the max node. Is this correct?
        var activeNode = UnderlyingNodes.Aggregate((curMax, x) =>
        {
            x.CalculateOutput(activationFunction);
            return curMax == null || x.Output > curMax.Output ? x : curMax;
        });
        Output = activeNode.Output;
        Weights = activeNode.Weights;
        BiasWeights = activeNode.BiasWeights;
    }

    public override void Initialise(Random rand, Func<Random, int, int, double> initialisationFunction, int nodeCount)
    {
        foreach(var node in UnderlyingNodes)
        {
            node.Initialise(rand, initialisationFunction, nodeCount);
        }
    }
}
