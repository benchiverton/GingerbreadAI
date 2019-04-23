namespace NeuralNetwork
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization.Formatters.Binary;
    using AI.Calculations;
    using Data;
    using Exceptions;

    public static class LayerExtensions
    {
        public static double[] GetResults(this Layer layer, double[] inputs)
        {
            PopulateAllResults(layer, inputs);
            return layer.Nodes.Select(n => n.Output).ToArray();
        }

        public static void PopulateResults(this Layer layer, double[] inputs)
        {
            PopulateAllResults(layer, inputs);
        }

        public static double GetResult(this Layer layer, int inputIndex, int outputIndex, double inputValue = 1)
        {
            PopulateResult(layer, inputIndex, outputIndex, inputValue);
            return layer.Nodes[outputIndex].Output;
        }

        public static void PopulateResult(this Layer layer, int inputIndex, int outputIndex, double inputValue)
        {
            PopulateSingleResults(layer, inputIndex, outputIndex, inputValue);
        }

        public static Layer DeepCopy(this Layer layer)
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);
                ms.Position = 0;
                return (Layer)formatter.Deserialize(ms);
            }
        }

        public static Layer CloneWithNodeReferences(this Layer layer)
        {
            return RecurseCloneWithNodeReferences(layer);
        }

        // Use this when multi-threading the same network
        public static Layer CloneNewWithWeightReferences(this Layer layer)
        {
            return RecurseCloneNewWithWeightReferences(layer);
        }

        public static void Save(this Layer layer, string location)
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, layer);
                using (var fileStream = File.Create(location))
                {
                    ms.WriteTo(fileStream);
                }
            }
        }

        #region Private Methods

        private static void PopulateSingleResults(Layer layer, int inputIndex, int outputIndex, double inputValue)
        {
            foreach (var previousLayer in layer.PreviousLayers)
            {
                var isInput = PopulateIndexedResults(previousLayer, inputIndex, inputValue);
                if (isInput) return;
            }

            var outputNode = layer.Nodes[outputIndex];
            outputNode.Output = 0;
            foreach (var previousNodeWeight in outputNode.Weights)
            {
                outputNode.Output += previousNodeWeight.Key.Output * previousNodeWeight.Value.Value;
            }
            foreach (var previousLayerWeight in outputNode.BiasWeights)
            {
                outputNode.Output += previousLayerWeight.Value.Value;
            }

            layer.Nodes[outputIndex].Output = NetworkCalculations.LogisticFunction(layer.Nodes[outputIndex].Output);
        }

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

        private static bool PopulateIndexedResults(Layer layer, int inputIndex, double inputValue)
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

        private static void HandleLayer(Layer layer, int inputIndex, double inputValue)
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

        private static Layer RecurseCloneWithNodeReferences(Layer layer)
        {
            var newLayer = new Layer
            {
                Nodes = new Node[layer.Nodes.Length],
                PreviousLayers = new Layer[layer.PreviousLayers.Length]
            };

            for (var i = 0; i < layer.Nodes.Length; i++)
            {
                var newNode = new Node
                {
                    Weights = new Dictionary<Node, Weight>(),
                    BiasWeights = new Dictionary<Layer, Weight>(),
                    Output = 0
                };

                foreach (var weightKey in layer.Nodes[i].Weights.Keys)
                {
                    newNode.Weights.Add(weightKey, new Weight(0));
                }

                foreach (var biasWeightKey in layer.Nodes[i].BiasWeights.Keys)
                {
                    newNode.BiasWeights.Add(biasWeightKey, new Weight(0));
                }

                newLayer.Nodes[i] = newNode;
            }

            for (var i = 0; i < layer.PreviousLayers.Length; i++)
            {
                newLayer.PreviousLayers[i] = RecurseCloneWithNodeReferences(layer.PreviousLayers[i]);
            }

            return newLayer;
        }

        private static Layer RecurseCloneNewWithWeightReferences(Layer layer)
        {
            if (!layer.PreviousLayers.Any())
            {
                var newInputLayer = new Layer()
                {
                    Name = $"{layer.Name}_CLONE",
                    Nodes = new Node[layer.Nodes.Length],
                    PreviousLayers = new Layer[0]
                };

                for (var i = 0; i < layer.Nodes.Length; i++)
                {
                    newInputLayer.Nodes[i] = new Node()
                    {
                        Weights = new Dictionary<Node, Weight>(),
                        BiasWeights = new Dictionary<Layer, Weight>(),
                        Output = 0
                    };
                }

                return newInputLayer;
            }

            var clonedPreviousLayers = new List<Layer>();
            foreach (var previousLayer in layer.PreviousLayers)
            {
                clonedPreviousLayers.Add(RecurseCloneNewWithWeightReferences(previousLayer));
            }

            var newLayer = new Layer()
            {
                Name = $"{layer.Name}_CLONE",
                PreviousLayers = clonedPreviousLayers.ToArray(),
                Nodes = new Node[layer.Nodes.Length]
            };

            for (var i = 0; i < layer.Nodes.Length; i++)
            {
                var newNode = new Node()
                {
                    Weights = new Dictionary<Node, Weight>(),
                    BiasWeights = new Dictionary<Layer, Weight>(),
                    Output = 0
                };

                for (var j = 0; j < layer.PreviousLayers.Length; j++)
                {
                    for (var k = 0; k < layer.PreviousLayers[j].Nodes.Length; k++)
                    {
                        newNode.Weights.Add(newLayer.PreviousLayers[j].Nodes[k], layer.Nodes[i].Weights[layer.PreviousLayers[j].Nodes[k]]);
                    }

                    newNode.BiasWeights.Add(newLayer.PreviousLayers[j], layer.Nodes[i].BiasWeights[layer.PreviousLayers[j]]);
                }

                newLayer.Nodes[i] = newNode;
            }

            return newLayer;
        }

        #endregion
    }
}
