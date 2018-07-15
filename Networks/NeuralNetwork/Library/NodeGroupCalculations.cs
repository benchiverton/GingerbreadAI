using NeuralNetwork.Data;
using NeuralNetwork.Data.Extensions;
using NeuralNetwork.Exceptions;

namespace NeuralNetwork.Library
{
    public static class NodeGroupCalculations
    {
        /// <summary>
        ///     Returns the result from this nodeGroup, using its previous groups.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="nodeGroup"></param>
        /// <returns></returns>
        public static void GetResult(NodeGroup nodeGroup, double[] inputs)
        {
            // this should happen if you have provided the incorrect amount of input for your layer
            if (nodeGroup.PreviousGroups.Length == 0
                && inputs.Length != nodeGroup.Nodes.Length)
            {
                throw new NodeNetworkException();
            }
            // this should only happen when you reach an input group
            if (nodeGroup.PreviousGroups.Length == 0)
            {
                nodeGroup.Outputs = inputs;
                return;
            }

            //ensure that the output array is clear
            System.Array.Clear(nodeGroup.Outputs, 0, nodeGroup.Outputs.Length);

            // select a group feeding into this one
            nodeGroup.PreviousGroups.Each((group, i) =>
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                GetResult(group, inputs);

                // iterate through Nodes in the current group
                for (var j = 0; j < nodeGroup.Nodes.Length; j++)
                {
                    // iterate through the outputs of the previous group, adding its weighted result to the results for this group
                    for (var k = 0; k < group.Outputs.Length; k++)
                    {
                        nodeGroup.Outputs[j] += group.Outputs[k] * nodeGroup.Nodes[j].Weights[i][k];
                    }

                    // add the bias for the previous group
                    nodeGroup.Outputs[j] += nodeGroup.Nodes[j].BiasWeights[i];
                }
            });

            // apply the logistic function to each of the results
            for (var i = 0; i < nodeGroup.Outputs.Length; i++)
            {
                nodeGroup.Outputs[i] = NodeCalculations.LogisticFunction(nodeGroup.Outputs[i]);
            }
        }
    }
}