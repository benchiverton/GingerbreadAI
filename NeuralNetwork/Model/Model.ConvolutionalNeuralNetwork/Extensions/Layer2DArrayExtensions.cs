using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Layer2DArrayExtensions
    {
        public static Filter2D[] Add2DConvolutionalLayer(this Layer2D[] inputs, int filterCount, int filterDimension)
        {
            var filters = new Filter2D[filterCount];
            for (var i = 0; i < filterCount; i++)
            {
                filters[i] = new Filter2D(inputs, filterDimension);
            }
            return filters;
        }
    }
}
