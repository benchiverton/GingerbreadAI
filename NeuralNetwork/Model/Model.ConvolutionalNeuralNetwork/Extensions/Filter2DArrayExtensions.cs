using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Filter2DArrayExtensions
    {
        public static void AddPooling(this Filter2D[] filters, int poolingDimension)
        {
            foreach (var filter in filters)
            {
                filter.AddPooling(poolingDimension);
            }
        }
    }
}
