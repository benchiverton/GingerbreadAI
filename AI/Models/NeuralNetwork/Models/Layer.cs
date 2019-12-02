using System;
using System.Text;

namespace NeuralNetwork.Models
{
    using System.Collections.Generic;
    using System.Linq;
    using AI.Calculations;
    using Exceptions;

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

            PopulateOutputs(inputs);
        }

        public void PopulateAllOutputs(Dictionary<Layer, double[]> inputs)
        {
            if (!PreviousLayers.Any())
            {
                PopulateInputLayersOutputs(inputs[this]);
                return;
            }

            PopulateOutputs(inputs);
        }

        public void PopulateIndexedOutputs(int inputIndex, int outputIndex, double inputValue)
        {
            foreach (var previousLayer in PreviousLayers)
            {
                PopulateIndexedOutputs(previousLayer, inputIndex, inputValue);
            }

            Nodes[outputIndex].PopulateOutput();
        }

        public void PopulateListWithInputLayers(List<Layer> inputLayerList)
        {
            if (!PreviousLayers.Any())
            {
                inputLayerList.Add(this);
            }
            else
            {
                foreach (var previousLayer in PreviousLayers)
                {
                    previousLayer.PopulateListWithInputLayers(inputLayerList);
                }
            }
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
                throw new NeuralNetworkException($"Input layer length ({Nodes.Length}) not equal to length of your inputs ({inputs.Length}).");

            var i = 0;
            foreach (var node in Nodes)
            {
                node.Output = inputs[i++];
            }
        }

        private void PopulateOutputs(double[] inputs)
        {
            foreach (var prevLayer in PreviousLayers)
            {
                prevLayer.PopulateAllOutputs(inputs);
            }

            foreach (var node in Nodes)
            {
                node.PopulateOutput();
            }
        }

        private void PopulateOutputs(Dictionary<Layer, double[]> inputs)
        {
            foreach (var prevLayer in PreviousLayers)
            {
                prevLayer.PopulateAllOutputs(inputs);
            }

            foreach (var node in Nodes)
            {
                node.PopulateOutput();
            }
        }

        private bool PopulateIndexedOutputs(Layer layer, int inputIndex, double inputValue)
        {
            if (!layer.PreviousLayers.Any())
            {
                layer.Nodes[inputIndex].Output = inputValue;
                return true;
            }

            HandleIndexedLayer(layer, inputIndex, inputValue);

            return false;
        }

        private void HandleIndexedLayer(Layer layer, int inputIndex, double inputValue)
        {
            foreach (var prevLayer in layer.PreviousLayers)
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                var isNextToInput = PopulateIndexedOutputs(prevLayer, inputIndex, inputValue);

                if (isNextToInput)
                {
                    foreach (var node in layer.Nodes)
                    {
                        node.Output = node.Weights[prevLayer.Nodes[inputIndex]].Value * prevLayer.Nodes[inputIndex].Output + node.BiasWeights[prevLayer].Value;
                        node.Output = NetworkCalculations.LogisticFunction(node.Output);
                    }
                    return;
                }
            }

            foreach (var node in layer.Nodes)
            {
                node.PopulateOutput();
            }
        }

        #endregion
    }
}