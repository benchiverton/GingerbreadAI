using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Library.Computations;
using Model.NeuralNetwork.Exceptions;

namespace Model.NeuralNetwork.Models
{
    [Serializable]
    public class Layer
    {
        /// <summary>
        ///     The name of the node.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        ///     An array of the nodes within this group.
        /// </summary>
        public Node[] Nodes { get; set; }

        /// <summary>
        ///     An array containing the NodeGroups that feed into this one.
        /// </summary>
        public Layer[] PreviousLayers { get; set; }

        public Layer()
        {
        }

        public Layer(string name, int nodeCount, Layer[] previousGroups)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            for (var i = 0; i < nodeCount; i++)
            {
                Nodes[i] = new Node(previousGroups);
            }
            PreviousLayers = previousGroups;
        }

        public void PopulateAllOutputs(double[] inputs)
        {
            if (!PreviousLayers.Any())
            {
                PopulateInputLayersOutputs(inputs);
                return;
            }

            foreach (var prevLayer in PreviousLayers)
            {
                prevLayer.PopulateAllOutputs(inputs);
            }

            foreach (var node in Nodes)
            {
                node.PopulateOutput();
            }
        }

        public void PopulateAllOutputs(Dictionary<Layer, double[]> inputs)
        {
            if (!PreviousLayers.Any())
            {
                PopulateInputLayersOutputs(inputs[this]);
                return;
            }

            foreach (var prevLayer in PreviousLayers)
            {
                prevLayer.PopulateAllOutputs(inputs);
            }

            foreach (var node in Nodes)
            {
                node.PopulateOutput();
            }
        }

        /// <summary>
        ///  Note: does not support multiple inputs
        /// </summary>
        public void PopulateIndexedOutput(int inputIndex, int outputIndex, double inputValue)
        {
            foreach (var previousLayer in PreviousLayers)
            {
                PopulateIndexedOutput(previousLayer, inputIndex, inputValue);
            }

            Nodes[outputIndex].PopulateOutput();
        }

        public string ToString(bool recurse = false, int layer = 0)
        {
            var indentation = "";
            for (var i = 0; i < layer; i++)
            {
                indentation += "    ";
            }

            var s = new StringBuilder($"{indentation}Node Group: {Name}; Node count: {Nodes.Length}\n");
            s.Append($"{indentation}Previous Groups:\n");

            layer++;
            foreach (var nodeGroup in PreviousLayers)
            {
                s.Append(recurse
                    ? nodeGroup.ToString(true, layer)
                    : $"{indentation}Node Group: {nodeGroup.Name}; Node count: {nodeGroup.Nodes.Length}\n");
            }
            return s.ToString();
        }

        #region Private methods


        private void PopulateInputLayersOutputs(double[] inputs)
        {
            if (Nodes.Length != inputs.Length)
            {
                throw new NeuralNetworkException($"Input layer length ({Nodes.Length}) not equal to length of your inputs ({inputs.Length}).");
            }

            var i = 0;
            foreach (var node in Nodes)
            {
                node.Output = inputs[i++];
            }
        }

        private bool PopulateIndexedOutput(Layer layer, int inputIndex, double inputValue)
        {
            if (!layer.PreviousLayers.Any())
            {
                layer.Nodes[inputIndex].Output = inputValue;
                return true;
            }

            var shouldPopulateAllOutputs = false;
            foreach (var prevLayer in layer.PreviousLayers)
            {
                var isNextToInput = PopulateIndexedOutput(prevLayer, inputIndex, inputValue);

                if (isNextToInput)
                {
                    var inputNode = prevLayer.Nodes[inputIndex];
                    foreach (var node in layer.Nodes)
                    {
                        node.Output = node.Weights[inputNode].Value * inputNode.Output + node.BiasWeights[prevLayer].Value;
                        node.Output = LogisticFunction.ComputeOutput(node.Output);
                    }
                }
                else
                {
                    // if not next to input, all outputs need loading - will break if multiple inputs
                    shouldPopulateAllOutputs = true;
                }
            }

            if (shouldPopulateAllOutputs)
            {
                foreach (var node in layer.Nodes)
                {
                    node.PopulateOutput();
                }
            }

            return false;
        }

        #endregion
    }
}