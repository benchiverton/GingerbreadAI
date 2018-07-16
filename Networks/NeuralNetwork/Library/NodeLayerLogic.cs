using NeuralNetwork.Data;
using NeuralNetwork.Exceptions;

namespace NeuralNetwork.Library
{
    using System.Linq;
    using Bens.WonderfulExtensions;

    public class NodeLayerLogic
    {
        public NodeLayer OutputLayer { get; set; }
        
        public double[] GetResults(double[] inputs)
        {
            PopulateResults(inputs);
            return OutputLayer.Outputs.Values.ToArray();
        }

        public void PopulateResults(double[] inputs)
        {
            PopulateResults(OutputLayer, inputs);
        }

        private static void PopulateResults(NodeLayer nodeLayer, double[] inputs)
        {
            // this should only happen when you reach an input group
            if (nodeLayer.PreviousGroups.Length == 0)
            {
                if (nodeLayer.Nodes.Length != inputs.Length)
                {
                    throw new NodeNetworkException();
                }
                nodeLayer.Nodes.Each((node, i) =>
                {
                    nodeLayer.Outputs[node] = inputs[i];
                });
                return;
            }

            //ensure that the output array is clear
            var outputKeys = nodeLayer.Outputs.Keys.ToList();
            foreach (var outputKey in outputKeys)
            {
                nodeLayer.Outputs[outputKey] = 0;
            }

            // select a group feeding into this one
            nodeLayer.PreviousGroups.Each((prevGroup, i) =>
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                PopulateResults(prevGroup, inputs);

                foreach (var node in nodeLayer.Nodes)
                {
                    foreach (var output in prevGroup.Outputs.Keys.ToList())
                    {
                        nodeLayer.Outputs[node] += prevGroup.Outputs[output] * node.Weights[prevGroup][output];
                    }
                }
            });

            // apply the logistic function to each of the results
            foreach (var output in nodeLayer.Outputs.Keys.ToList())
            {
                nodeLayer.Outputs[output] = NodeCalculations.LogisticFunction(nodeLayer.Outputs[output]);
            }
        }
    }
}
