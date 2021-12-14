using System;
using System.Collections.Generic;
using System.Linq;

namespace GingerbreadAI.Model.NeuralNetwork.Models;

public class Node
{
    public Node()
    {
    }

    public Node(IReadOnlyList<Layer> nodeGroupPrev, bool addBiasWeights)
    {
        foreach (var prevNodeLayer in nodeGroupPrev)
        {
            foreach (var node in prevNodeLayer.Nodes)
            {
                Weights.Add(node, new Weight(0));
            }
        }

        if (addBiasWeights)
        {
            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                BiasWeights.Add(prevNodeLayer, new Weight(0));
            }
        }
    }

    /// <summary>
    /// The weights, with reference to the layer & node the value id being mapped from
    /// </summary>
    public Dictionary<Node, Weight> Weights { get; protected set; } = new Dictionary<Node, Weight>();

    /// <summary>
    /// The bias weights, with reference to the layer the value is mapped from
    /// </summary>
    public Dictionary<Layer, Weight> BiasWeights { get; protected set; } = new Dictionary<Layer, Weight>();

    /// <summary>
    /// The output of the node from the last results calculation.
    /// </summary>
    public double Output { get; set; }

    public virtual void CalculateOutput(Func<double, double> activationFunction)
    {
        var output = 0d;

        foreach (var weight in Weights)
        {
            output += weight.Key.Output * weight.Value.Value;
        }
        foreach (var weight in BiasWeights)
        {
            output += weight.Value.Value;
        }

        Output = activationFunction.Invoke(output);
    }

    public virtual void Initialise(Random rand, Func<Random, int, int, double> initialisationFunction, int nodeCount)
    {
        var feedingNodes = Weights.Count;
        foreach (var prevNode in Weights.Keys.ToList())
        {
            Weights[prevNode].Adjust(initialisationFunction.Invoke(rand, feedingNodes, nodeCount));
        }
        var biasWeightKeys = new List<Layer>(BiasWeights.Keys.ToList());
        foreach (var biasWeightKey in biasWeightKeys)
        {
            BiasWeights[biasWeightKey].Adjust(initialisationFunction.Invoke(rand, feedingNodes, nodeCount));
        }
    }
}
