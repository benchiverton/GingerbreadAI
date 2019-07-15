using System;
using System.Text;

namespace NeuralNetwork.Models
{
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
            // default constructor
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
            foreach (var node in Nodes)
            {
                node.Output = 0;
            }
        }

        public void PopulateResults(double[] inputs)
        {
            // this should only happen when you reach an input group
            if (!PreviousLayers.Any())
            {
                HandleInputLayer(this, inputs);
                return;
            }

            // ensure that the output array is clear
            foreach (var node in Nodes)
            {
                node.Output = 0;
            }

            HandleLayer(this, inputs);

            // apply the logistic function to each of the results
            foreach (var node in Nodes)
            {
                node.Output = NetworkCalculations.LogisticFunction(node.Output);
            }
        }

        public void PopulateResult(int inputIndex, int outputIndex, double inputValue)
        {
            foreach (var previousLayer in PreviousLayers)
            {
                var isInput = PopulateIndexedResults(previousLayer, inputIndex, inputValue);
                if (isInput) return;
            }

            var outputNode = Nodes[outputIndex];
            outputNode.Output = 0;
            foreach (var previousNodeWeight in outputNode.Weights)
            {
                outputNode.Output += previousNodeWeight.Key.Output * previousNodeWeight.Value.Value;
            }
            foreach (var previousLayerWeight in outputNode.BiasWeights)
            {
                outputNode.Output += previousLayerWeight.Value.Value;
            }

            Nodes[outputIndex].Output = NetworkCalculations.LogisticFunction(Nodes[outputIndex].Output);
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


        private void HandleInputLayer(Layer nodeLayer, double[] inputs)
        {
            if (nodeLayer.Nodes.Length != inputs.Length)
                throw new NeuralNetworkException($"Input layer length ({nodeLayer.Nodes.Length}) not equal to length of your inputs ({inputs.Length}).");

            var i = 0;
            foreach (var node in nodeLayer.Nodes)
            {
                node.Output = inputs[i++];
            }
        }

        private void HandleLayer(Layer nodeLayer, double[] inputs)
        {
            foreach (var prevLayer in nodeLayer.PreviousLayers)
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                prevLayer.PopulateResults(inputs);

                foreach (var node in nodeLayer.Nodes)
                {
                    foreach (var prevNode in prevLayer.Nodes)
                    {
                        node.Output += prevNode.Output * node.Weights[prevNode].Value;
                    }

                    node.Output += node.BiasWeights[prevLayer].Value;
                }
            }
        }

        private bool PopulateIndexedResults(Layer layer, int inputIndex, double inputValue)
        {
            if (!layer.PreviousLayers.Any())
            {
                layer.Nodes[inputIndex].Output = inputValue;
                return true;
            }

            foreach (var node in layer.Nodes)
            {
                node.Output = 0;
            }

            HandleLayer(layer, inputIndex, inputValue);

            return false;
        }

        private void HandleLayer(Layer layer, int inputIndex, double inputValue)
        {
            foreach (var prevLayer in layer.PreviousLayers)
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                var isNextToInput = PopulateIndexedResults(prevLayer, inputIndex, inputValue);

                if (isNextToInput)
                {
                    foreach (var node in layer.Nodes)
                    {
                        node.Output = node.Weights[prevLayer.Nodes[inputIndex]].Value * prevLayer.Nodes[inputIndex].Output + node.BiasWeights[prevLayer].Value;
                        node.Output = NetworkCalculations.LogisticFunction(node.Output);
                    }
                }
                else
                {
                    foreach (var node in layer.Nodes)
                    {
                        foreach (var prevNode in prevLayer.Nodes)
                        {
                            node.Output += prevNode.Output * node.Weights[prevNode].Value;
                        }

                        node.Output += node.BiasWeights[prevLayer].Value;
                        node.Output = NetworkCalculations.LogisticFunction(node.Output);
                    }
                }
            };
        }

        #endregion
    }
}