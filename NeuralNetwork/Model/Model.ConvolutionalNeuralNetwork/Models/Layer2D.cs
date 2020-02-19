using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Layer2D : Layer
    {
        public (int height, int width) Dimensions { get; }

        public Layer2D((int height, int width) dimensions, Layer[] previousGroups)
        : base(dimensions.height * dimensions.width, previousGroups)
        {
            Dimensions = dimensions;
        }
    }
}
