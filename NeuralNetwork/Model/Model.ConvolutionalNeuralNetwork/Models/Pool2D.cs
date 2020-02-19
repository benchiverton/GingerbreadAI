using System.Collections.Generic;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Pool2D : Layer
    {
        public Pool2D(Filter2D filter, int dimension)
        {
            PreviousLayers = new Layer[] { filter };

            var nodes = new List<Node>();

            var (height, width) = (filter.PreviousLayers[0] as Layer2D).Dimensions;
            for (var i = 0; i < height - filter.Dimension - dimension + 2; i++)
            {
                for (var j = 0; j < width - filter.Dimension - dimension + 2; j++)
                {
                    var nodeWeights = new Dictionary<Node, Weight>();
                    for (var k = 0; k < dimension; k++) // across
                    {
                        for (var l = 0; l < dimension; l++) // down
                        {
                            var nodePosition = j + l + (i + k) * (width - filter.Dimension + 1);
                            nodeWeights.Add(filter.Nodes[nodePosition], new Pool2DWeight(dimension));
                        }
                    }
                    nodes.Add(new Node
                    {
                        Weights = nodeWeights
                    });
                }
            }

            Nodes = nodes.ToArray();
        }
    }

    public class Pool2DWeight : Weight
    {
        public Pool2DWeight(int dimensions) : base(0)
        {
            _dimensions = dimensions;
        }

        private readonly int _dimensions;

        public override double Value => 1d / _dimensions;
    }
}
