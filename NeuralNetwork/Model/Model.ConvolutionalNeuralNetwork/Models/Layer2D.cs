using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Layer2D : Layer
    {
        public (int height, int width) Dimensions { get; }

        public Layer2D((int height, int width) dimensions, Layer[] previousGroups, 
            ActivationFunctionType activationFunctionType, InitialisationFunctionType initialisationFunctionType)
        : base(dimensions.height * dimensions.width, previousGroups, activationFunctionType, initialisationFunctionType)
        {
            Dimensions = dimensions;
        }
    }
}
