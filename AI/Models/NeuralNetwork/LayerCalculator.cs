namespace NeuralNetwork
{
    using System.Linq;
    using AI.Calculations;
    using Data;
    using Exceptions;

    public class LayerCalculator
    {
        public Layer OutputLayer { get; }

        public LayerCalculator(Layer outputLayer)
        {
            OutputLayer = outputLayer;
        }

        public double[] GetResults(double[] inputs)
        {
            PopulateResults(inputs);
            return OutputLayer.Nodes.Select(n => n.Output).ToArray();
        }

        public void PopulateResults(double[] inputs)
        {
            PopulateResults(OutputLayer, inputs);
        }

        private static void PopulateResults(Layer nodeLayer, double[] inputs)
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
                PopulateResults(prevLayer, inputs);

                foreach (var node in nodeLayer.Nodes)
                {
                    foreach (var prevNode in prevLayer.Nodes)
                    {
                        node.Output += prevNode.Output * node.Weights[prevNode].Value;
                    }

                    node.Output += node.BiasWeights[prevLayer].Value;
                }
            };
        }
    }
}
