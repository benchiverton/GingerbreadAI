namespace NeuralNetwork
{
    using System.Linq;
    using Bens.WonderfulLibrary.Calculations;
    using Bens.WonderfulLibrary.Extensions;
    using Data;
    using Exceptions;

    public class LayerComputor
    {
        public Layer OutputLayer { get; set; }

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
            if (nodeLayer.PreviousLayers.Length == 0)
            {
                if (nodeLayer.Nodes.Length != inputs.Length)
                {
                    throw new NeuralNetworkException();
                }
                nodeLayer.Nodes.Each((node, i) =>
                {
                    node.Output = inputs[i];
                });
                return;
            }

            //ensure that the output array is clear
            foreach (var node in nodeLayer.Nodes)
            {
                node.Output = 0;
            }

            // select a group feeding into this one
            nodeLayer.PreviousLayers.Each((prevGroup, i) =>
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                PopulateResults(prevGroup, inputs);

                foreach (var node in nodeLayer.Nodes)
                {
                    foreach (var prevNode in prevGroup.Nodes)
                    {
                        node.Output += prevNode.Output * node.Weights[prevNode];
                    }

                    node.Output += node.BiasWeights[prevGroup];
                }
            });

            // apply the logistic function to each of the results
            foreach (var node in nodeLayer.Nodes)
            {
                node.Output = NetworkCalculations.LogisticFunction(node.Output);
            }
        }
    }
}
