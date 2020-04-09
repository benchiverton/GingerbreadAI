using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Initialisers;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Layer1DArrayExtensions
    {
        public static Filter1D[] Add2DConvolutionalLayer(this Layer1D[] inputs, int filterCount, int filterSize,
            ActivationFunctionType activationFunction, InitialisationFunctionType initialisationFunction)
        {
            var filters = new Filter1D[filterCount];
            for (var i = 0; i < filterCount; i++)
            {
                filters[i] = new Filter1D(inputs, filterSize, activationFunction, initialisationFunction);
            }
            return filters;
        }
    }
}
