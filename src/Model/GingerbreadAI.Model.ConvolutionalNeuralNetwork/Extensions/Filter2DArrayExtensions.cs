using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;

public static class Filter2DArrayExtensions
{
    public static void AddPooling(this Filter2D[] filters, (int height, int width) poolingDimensions)
    {
        foreach (var filter in filters)
        {
            filter.AddPooling(poolingDimensions);
        }
    }
}
