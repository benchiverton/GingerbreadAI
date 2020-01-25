using System.Collections.Generic;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Filter : Layer
    {
        public Filter(Layer[] previousLayers, int prvLayersHeight, int prvLayersWidth, int filterDimension)
        {
            PreviousLayers = previousLayers;
            
            var nodeCount = (prvLayersHeight - filterDimension) * (prvLayersWidth - filterDimension);
            Nodes = new Node[nodeCount];

            for (var i = 0; i < nodeCount; i++)
            {
                var linkedPrvNodes = new List<Node>();
                for (var j = 0; j < filterDimension; j++)
                {
                    foreach (var layer in previousLayers)
                    {
                        for (var k = 0; k < filterDimension; k++)
                        {
                            linkedPrvNodes.Add(layer.Nodes[j * prvLayersWidth + k]);
                        }
                    }
                }
                Nodes[i] = new Node(linkedPrvNodes);
            }
        }
    }
}
