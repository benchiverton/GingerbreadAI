using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;

namespace GingerbreadAI.Model.NeuralNetwork.Models;

public class Layer
{
    private readonly Guid _id = Guid.NewGuid();

    private IReadOnlyList<Node> _nodesWrapper;
    private IReadOnlyList<Layer> _previousLayersWrapper;

    private Node[] _nodes;
    private Layer[] _previousLayers;

    private ActivationFunctionType _activationFunctionType;
    private InitialisationFunctionType _initialisationFunctionType;

    public Layer(
        Node[] nodes,
        Layer[] previousLayers,
        ActivationFunctionType activationFunctionType = ActivationFunctionType.RELU,
        InitialisationFunctionType initialisationFunctionType = InitialisationFunctionType.HeEtAl)
    {
        Nodes = nodes;
        PreviousLayers = previousLayers;
        ActivationFunctionType = activationFunctionType;
        InitialisationFunctionType = initialisationFunctionType;
    }

    public Layer(int nodeCount, Layer[] previousGroups, ActivationFunctionType activationFunctionType, InitialisationFunctionType initialisationFunctionType, bool addBiasWeights = true)
    {
        ActivationFunctionType = activationFunctionType;
        InitialisationFunctionType = initialisationFunctionType;
        PreviousLayers = previousGroups;

        var nodes = new Node[nodeCount];
        for (var i = 0; i < nodeCount; i++)
        {
            nodes[i] = new Node(previousGroups, addBiasWeights);
        }
        Nodes = nodes;
    }

    /// <summary>
    /// An array of the nodes within this layer.
    /// </summary>
    public IReadOnlyList<Node> Nodes
    {
        get => _nodesWrapper;
        set
        {
            _nodes = value.ToArray();
            _nodesWrapper = new ReadOnlyCollection<Node>(_nodes);
        }
    }

    /// <summary>
    /// An array containing the layers that feed into this one.
    /// </summary>
    public IReadOnlyList<Layer> PreviousLayers
    {
        get => _previousLayersWrapper;
        set
        {
            _previousLayers = value.ToArray();
            _previousLayersWrapper = new ReadOnlyCollection<Layer>(_previousLayers);
        }
    }

    /// <summary>
    /// The function used to calculate the output given the aggregated input.
    /// </summary>
    public Func<double, double> ActivationFunction { get; private set; }

    /// <summary>
    /// The differential of the function used to calculate the output given the aggregated input.
    /// </summary>
    public Func<double, double> ActivationFunctionDifferential { get; private set; }

    /// <summary>
    /// The function used to calculate the initial weights of the layer.
    /// </summary>
    public Func<Random, int, int, double> InitialisationFunction { get; private set; }

    public ActivationFunctionType ActivationFunctionType
    {
        get => _activationFunctionType;
        set
        {
            _activationFunctionType = value;
            (ActivationFunction, ActivationFunctionDifferential) = ActivationFunctionResolver.ResolveActivationFunctions(value);
        }
    }

    public InitialisationFunctionType InitialisationFunctionType
    {
        get => _initialisationFunctionType;
        set
        {
            _initialisationFunctionType = value;
            InitialisationFunction = InitialisationFunctionResolver.ResolveInitialisationFunctions(value);
        }
    }

    public void CalculateOutputs(double[] inputs) => CalculateOutputs(inputs, new List<Guid>());

    public void CalculateOutputs(Dictionary<Layer, double[]> inputs) => CalculateOutputs(inputs, new List<Guid>());

    /// <summary>
    ///  Note: does not support multiple inputs
    /// </summary>
    public void CalculateIndexedOutput(int inputIndex, int outputIndex, double inputValue)
    {
        foreach (var previousLayer in PreviousLayers)
        {
            previousLayer.CalculateIndexedOutput(inputIndex, inputValue);
        }

        Nodes[outputIndex].CalculateOutput(ActivationFunction);
    }

    #region Private methods

    private void CalculateOutputs(double[] inputs, ICollection<Guid> processedLayers)
    {
        if (processedLayers.Contains(_id))
        {
            return;
        }

        processedLayers.Add(_id);
        if (!PreviousLayers.Any())
        {
            SetOutputs(inputs);
            return;
        }

        foreach (var prevLayer in PreviousLayers)
        {
            prevLayer.CalculateOutputs(inputs, processedLayers);
        }

        foreach (var node in Nodes)
        {
            node.CalculateOutput(ActivationFunction);
        }
    }

    private void CalculateOutputs(IReadOnlyDictionary<Layer, double[]> inputs, ICollection<Guid> processedLayers)
    {
        if (processedLayers.Contains(_id))
        {
            return;
        }

        processedLayers.Add(_id);
        if (!PreviousLayers.Any())
        {
            SetOutputs(inputs[this]);
            return;
        }

        foreach (var prevLayer in PreviousLayers)
        {
            prevLayer.CalculateOutputs(inputs, processedLayers);
        }

        foreach (var node in Nodes)
        {
            node.CalculateOutput(ActivationFunction);
        }
    }

    private void SetOutputs(double[] outputs)
    {
        if (Nodes.Count != outputs.Length)
        {
            throw new ArgumentException($"Layer length ({Nodes.Count}) not equal to length of your output array ({outputs.Length}).");
        }

        var i = 0;
        foreach (var node in Nodes)
        {
            node.Output = outputs[i++];
        }
    }

    private bool CalculateIndexedOutput(int inputIndex, double inputValue)
    {
        if (!PreviousLayers.Any())
        {
            Nodes[inputIndex].Output = inputValue;
            return true;
        }

        var shouldPopulateAllOutputs = false;
        foreach (var prevLayer in PreviousLayers)
        {
            shouldPopulateAllOutputs |= !prevLayer.CalculateIndexedOutput(inputIndex, inputValue);
            if (shouldPopulateAllOutputs)
            {
                continue;
            }

            var inputNode = prevLayer.Nodes[inputIndex];
            foreach (var node in Nodes)
            {
                var output = node.Weights[inputNode].Value * inputNode.Output;

                if (node.BiasWeights.TryGetValue(prevLayer, out var biasWeight))
                {
                    output += biasWeight.Value;
                }

                node.Output = ActivationFunction(output);
            }
        }

        if (shouldPopulateAllOutputs)
        {
            foreach (var node in Nodes)
            {
                node.CalculateOutput(ActivationFunction);
            }
        }

        return false;
    }

    #endregion
}
