namespace NeuralNetwork
{
    using System.Linq;
    using AI.Calculations;
    using Data;
    using Exceptions;

    public class OutputCalculator
    {
        public Layer OutputLayer { get; }

        public OutputCalculator(Layer outputLayer)
        {
            OutputLayer = outputLayer;
        }

        public double[] GetResults(double[] inputs)
        {
            PopulateResults(inputs);
            return OutputLayer.Nodes.Select(n => n.Output).ToArray();
        }

        public double GetResult(int inputIndex, int outputIndex, double inputValue = 1)
        {
            PopulateResults(inputIndex, outputIndex, inputValue);
            return OutputLayer.Nodes[outputIndex].Output;
        }

        public void PopulateResults(double[] inputs)
        {
            PopulateAllResults(OutputLayer, inputs);
        }

        public void PopulateResults(int inputIndex, int outputIndex, double inputValue)
        {
            foreach (var previousLayer in OutputLayer.PreviousLayers)
            {
                var isInput = PopulateIndexedResults(previousLayer, inputIndex, outputIndex, inputValue);
                if (isInput) return;
            }

            var outputNode = OutputLayer.Nodes[outputIndex];
            outputNode.Output = 0;
            foreach (var previousNodeWeight in outputNode.Weights)
            {
                outputNode.Output += previousNodeWeight.Key.Output * previousNodeWeight.Value.Value;
            }
            foreach(var previousLayerWeight in outputNode.BiasWeights)
            {
                outputNode.Output += previousLayerWeight.Value.Value;
            }

            OutputLayer.Nodes[outputIndex].Output = NetworkCalculations.LogisticFunction(OutputLayer.Nodes[outputIndex].Output);
        }

        #region PopulateAllResults

        private static void PopulateAllResults(Layer nodeLayer, double[] inputs)
        {
            // this should only happen when you reach an input group
            if (!nodeLayer.PreviousLayers.Any())
            {
                HandleInputLayer(nodeLayer, inputs);
                return;
            }

            // ensure that the output array is clear
            foreach (var node in nodeLayer.Nodes)
            {
                node.Output = 0;
            }

            HandleLayer(nodeLayer, inputs);

            // apply the logistic function to each of the results
            foreach (var node in nodeLayer.Nodes)
            {
                node.Output = NetworkCalculations.LogisticFunction(node.Output);
            }
        }

        private static void HandleInputLayer(Layer nodeLayer, double[] inputs)
        {
            if (nodeLayer.Nodes.Length != inputs.Length)
                throw new NeuralNetworkException($"Input layer length ({nodeLayer.Nodes.Length}) not equal to length of your inputs ({inputs.Length}).");

            var i = 0;
            foreach (var node in nodeLayer.Nodes)
            {
                node.Output = inputs[i++];
            }
        }

        private static void HandleLayer(Layer nodeLayer, double[] inputs)
        {
            foreach (var prevLayer in nodeLayer.PreviousLayers)
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                PopulateAllResults(prevLayer, inputs);

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

        #endregion

        #region PopulateIndexedResults

        private static bool PopulateIndexedResults(Layer layer, int inputIndex, int outputIndex, double inputValue)
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

            HandleLayer(layer, inputIndex, outputIndex, inputValue);

            return false;
        }

        private static void HandleLayer(Layer layer, int inputIndex, int outputIndex, double inputValue)
        {
            foreach (var prevLayer in layer.PreviousLayers)
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                var isNextToInput = PopulateIndexedResults(prevLayer, inputIndex, outputIndex, inputValue);

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
