using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Data;
using NeuralNetwork.Data.Extensions;

namespace NeuralNetwork.Library
{
    public class NodeLayerCalculations
    {
        /// <summary>
        ///     Returns the result from this nodeLayer, using its previous layers.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="nodeLayer"></param>
        /// <returns></returns>
        public static double[] GetResult(double[] inputs, NodeLayer nodeLayer)
        {
            // this should only happen when you reach an input layer
            if (nodeLayer.PreviousLayers == null)
                return inputs;
            // we have a result for each node, so I initialise the result array here
            var results = new double[nodeLayer.Nodes.Length];
            // select a layer feeding into this one
            nodeLayer.PreviousLayers.Each((layer, i) =>
            {
                // gets the results of the layer selected above (the 'previous layer'), which are the inputs for this layer
                var layerInputs = GetResult(inputs, layer);

                // iterate through Nodes in the current layer
                for (var j = 0; j < nodeLayer.Nodes.Length; j++)
                {
                    // iterate through the outputs of the previous layer, adding its weighted result to the results for this layer
                    for (var k = 0; k < layerInputs.Length; k++)
                        results[j] += layerInputs[k] * nodeLayer.Nodes[j].Weights[i][k];

                    // add the bias for the previous layer
                    results[j] += nodeLayer.Nodes[j].BiasWeights[i];
                }
            });

            // apply the logistic function to each of the results
            for (var i = 0; i < results.Length; i++)
                results[i] = NodeCalculations.LogisticFunction(results[i]);

            return results;
        }
    }
}
