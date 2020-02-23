using System;
using System.Collections.Generic;
using System.Linq;
using Library.Computations;

namespace Model.NeuralNetwork.Models
{
    public class Layer
    {
        /// <summary>
        ///     An array of the nodes within this layer.
        /// </summary>
        public Node[] Nodes { get; set; }

        /// <summary>
        ///     An array containing the layers that feed into this one.
        /// </summary>
        public Layer[] PreviousLayers { get; set; }

        public Layer()
        {
        }

        public Layer(int nodeCount, Layer[] previousGroups)
        {
            Nodes = new Node[nodeCount];
            PreviousLayers = previousGroups;

            for (var i = 0; i < nodeCount; i++)
            {
                Nodes[i] = new Node(previousGroups);
            }
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

        #region Private methods


        private void PopulateInputLayersOutputs(double[] inputs)
        {
            if (Nodes.Length != inputs.Length)
            {
                throw new ArgumentException($"Input layer length ({Nodes.Length}) not equal to length of your inputs ({inputs.Length}).");
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