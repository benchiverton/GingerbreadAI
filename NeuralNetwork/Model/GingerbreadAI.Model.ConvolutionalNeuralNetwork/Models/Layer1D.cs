using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Layer1D : Layer
    {
        public int Size { get; }

        public Layer1D(int size, Layer[] previousGroups, 
            ActivationFunctionType activationFunctionType, InitialisationFunctionType initialisationFunctionType)
        : base(size, previousGroups, activationFunctionType, initialisationFunctionType)
        {
            Size = size;
        }
    }
}
