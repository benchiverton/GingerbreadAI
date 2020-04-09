using Model.ConvolutionalNeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Filter1DArrayExtensions
    {
        public static void AddPooling(this Filter1D[] filters, int poolingDimension)
        {
            foreach (var filter in filters)
            {
                filter.AddPooling(poolingDimension);
            }
        }
    }
}
