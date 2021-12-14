using System;
using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Test.Extensions;

public class Filter2DArrayExtensionsShould
{
    [Fact]
    public void CorrectlyPoolToSingle()
    {
        // Input:
        // 0  1  2
        // 3  4  5
        // 6  7  8
        //
        // Filter:
        // 0,1,3,4  1,2,4,5
        // 3,4,6,7  4,5,7,8
        //
        // Pooling:
        // max(0,1,3,4; 1,2,4,5; 3,4,6,7; 4,5,7,8)
        var input = new Layer2D((3, 3), Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
        var filter = new Filter2D(new[] { input }, (2, 2), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);

        filter.AddPooling((2, 2));

        var node = Assert.Single(filter.Nodes);
        var pooledNode = Assert.IsType<PooledNode>(node);
        Assert.Equal(4, pooledNode.UnderlyingNodes.Count);
    }

    [Fact]
    public void CorrectlyPoolToMultiple()
    {
        // Input:
        // 0  1  2  3  4
        // 5  6  7  8  9
        // 10 11 12 13 14
        // 15 16 17 18 19
        // 20 21 22 23 24
        //
        // Filter:
        // 0,1,5,6     1,2,6,7     2,3,7,8     3,4,8,9
        // 5,6,10,11   6,7,11,12   7,8,12,13   8,9,13,14
        // 10,11,15,16 11,12,16,17 12,13,17,18 13,14,18,19
        // 15,16,20,21 16,17,21,22 17,18,22,23 18,19,23,24
        //
        // Pooling:
        // Node[0]: max(0,1,5,6;     1,2,6,7;     5,6,10,11;   6,7,11,12)
        // Node[1]: max(2,3,7,8;     3,4,8,9;    7,8,12,13;   8,9,13,14)
        // Node[2]: max(10,11,15,16; 11,12,16,17; 15,16,20,21; 16,17,21,22
        // Node[3]: max(12,13,17,18; 13,14,18,19; 17,18,22,23; 18,19,23,24)
        var input = new Layer2D((5, 5), Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
        var filter = new Filter2D(new[] { input }, (2, 2), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);

        filter.AddPooling((2, 2));

        Assert.Equal(4, filter.Nodes.Count);

        // assert inputs to nodes are as expected (index + occurences)
        var node0 = Assert.IsType<PooledNode>(filter.Nodes[0]);
        var node0inputs = node0.UnderlyingNodes
            .Aggregate(new List<Node>(), (current, next) => current.Concat(next.Weights.Keys).ToList())
            .GroupBy(n => n)
            .ToDictionary(n => n.Key);
        foreach (var n in new[] {
            (0, 1),
            (1, 2),
            (5, 2),
            (6, 4),
            (2, 1),
            (7, 2),
            (10, 1),
            (11, 2),
            (12, 1),
        })
        {
            Assert.Equal(n.Item2, node0inputs[input.Nodes[n.Item1]].Count());
        }
        var node1 = Assert.IsType<PooledNode>(filter.Nodes[1]);
        var node1inputs = node1.UnderlyingNodes
            .Aggregate(new List<Node>(), (current, next) => current.Concat(next.Weights.Keys).ToList())
            .GroupBy(n => n)
            .ToDictionary(n => n.Key);
        foreach (var n in new[] {
            (2, 1),
            (3, 2),
            (7, 2),
            (8, 4),
            (4, 1),
            (9, 2),
            (12, 1),
            (13, 2),
            (14, 1),
        })
        {
            Assert.Equal(n.Item2, node1inputs[input.Nodes[n.Item1]].Count());
        }
        var node2 = Assert.IsType<PooledNode>(filter.Nodes[2]);
        var node2inputs = node2.UnderlyingNodes
            .Aggregate(new List<Node>(), (current, next) => current.Concat(next.Weights.Keys).ToList())
            .GroupBy(n => n)
            .ToDictionary(n => n.Key);
        foreach (var n in new[] {
            (10, 1),
            (11, 2),
            (15, 2),
            (16, 4),
            (12, 1),
            (17, 2),
            (20, 1),
            (21, 2),
            (22, 1),
        })
        {
            Assert.Equal(n.Item2, node2inputs[input.Nodes[n.Item1]].Count());
        }
        var node3 = Assert.IsType<PooledNode>(filter.Nodes[3]);
        var node3inputs = node3.UnderlyingNodes
            .Aggregate(new List<Node>(), (current, next) => current.Concat(next.Weights.Keys).ToList())
            .GroupBy(n => n)
            .ToDictionary(n => n.Key);
        foreach (var n in new[] {
            (12, 1),
            (13, 2),
            (17, 2),
            (18, 4),
            (14, 1),
            (19, 2),
            (22, 1),
            (23, 2),
            (24, 1),
        })
        {
            Assert.Equal(n.Item2, node3inputs[input.Nodes[n.Item1]].Count());
        }
    }
}
