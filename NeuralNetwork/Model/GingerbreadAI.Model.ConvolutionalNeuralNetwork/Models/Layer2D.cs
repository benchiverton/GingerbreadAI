using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Layer2D : Layer
    {
        public (int height, int width) Shape { get; }

        public Layer2D((int height, int width) shape, Layer[] previousGroups, 
            ActivationFunctionType activationFunctionType, InitialisationFunctionType initialisationFunctionType)
        : base(shape.height * shape.width, previousGroups, activationFunctionType, initialisationFunctionType)
        {
            Shape = shape;
        }
    }
}
