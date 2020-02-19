using System;
using System.Collections.Generic;
using System.Linq;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Filter2DArrayExtensions
    {
        public static IEnumerable<Pool2D> AddPooling(this Filter2D[] filters, int poolingDimension)
        {
            return filters.Select(filter => new Pool2D(filter, poolingDimension));
        }
    }
}
