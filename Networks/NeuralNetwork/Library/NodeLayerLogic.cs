using NeuralNetwork.Data;
using NeuralNetwork.Data.Extensions;
using NeuralNetwork.Exceptions;

namespace NeuralNetwork.Library
{
    public class NodeLayerLogic
    {
        public NodeLayer OutputLayer { get; set; }

        /// <summary>
        ///     Returns the result from this nodeGroup, using its previous groups.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public void PopulateResults(double[] inputs)
        {
            // this should happen if you have provided the incorrect amount of input for your layer
            if (OutputLayer.PreviousGroups.Length == 0
                && inputs.Length != OutputLayer.Nodes.Length)
            {
                throw new NodeNetworkException();
            }
            
            PopulateResults(OutputLayer, inputs);
        }

        public double[] GetResults(double[] inputs)
        {
            PopulateResults(inputs);
            return OutputLayer.Outputs;
        }

        public void PopulateResults(NodeLayer nodeLayer, double[] inputs)
        {
            // this should only happen when you reach an input group
            if (nodeLayer.PreviousGroups.Length == 0)
            {
                nodeLayer.Outputs = inputs;
                return;
            }

            //ensure that the output array is clear
            System.Array.Clear(nodeLayer.Outputs, 0, nodeLayer.Outputs.Length);

            // select a group feeding into this one
            nodeLayer.PreviousGroups.Each((prevGroup, i) =>
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                PopulateResults(prevGroup, inputs);

                // iterate through Nodes in the current group
                for (var j = 0; j < nodeLayer.Nodes.Length; j++)
                {
                    // iterate through the outputs of the previous group, adding its weighted result to the results for this group
                    for (var k = 0; k < prevGroup.Outputs.Length; k++)
                    {
                        nodeLayer.Outputs[j] += prevGroup.Outputs[k] * nodeLayer.Nodes[j].Weights[i][k];
                    }

                    // add the bias for the previous group
                    nodeLayer.Outputs[j] += nodeLayer.Nodes[j].BiasWeights[i];
                }
            });

            // apply the logistic function to each of the results
            for (var i = 0; i < nodeLayer.Outputs.Length; i++)
            {
                nodeLayer.Outputs[i] = NodeCalculations.LogisticFunction(nodeLayer.Outputs[i]);
            }
        }
    }
}
