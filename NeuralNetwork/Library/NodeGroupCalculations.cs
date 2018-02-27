using NeuralNetwork.Data;
using NeuralNetwork.Data.Extensions;

namespace NeuralNetwork.Library
{
    public class NodeGroupCalculations
    {
        /// <summary>
        ///     Returns the result from this nodeGroup, using its previous groups.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="nodeGroup"></param>
        /// <returns></returns>
        public static double[] GetResult(double[] inputs, NodeGroup nodeGroup)
        {
            // this should only happen when you reach an input group
            if (nodeGroup.PreviousGroups == null)
                return inputs;
            // we have a result for each node, so I initialise the result array here
            var results = new double[nodeGroup.Nodes.Length];
            // select a group feeding into this one
            nodeGroup.PreviousGroups.Each((group, i) =>
            {
                // gets the results of the group selected above (the 'previous group'), which are the inputs for this group
                var groupInputs = GetResult(inputs, group);

                // iterate through Nodes in the current group
                for (var j = 0; j < nodeGroup.Nodes.Length; j++)
                {
                    // iterate through the outputs of the previous group, adding its weighted result to the results for this group
                    for (var k = 0; k < groupInputs.Length; k++)
                        results[j] += groupInputs[k] * nodeGroup.Nodes[j].Weights[i][k];

                    // add the bias for the previous group
                    results[j] += nodeGroup.Nodes[j].BiasWeights[i];
                }
            });

            // apply the logistic function to each of the results
            for (var i = 0; i < results.Length; i++)
                results[i] = NodeCalculations.LogisticFunction(results[i]);

            return results;
        }
    }
}
