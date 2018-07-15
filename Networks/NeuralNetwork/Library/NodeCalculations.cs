using System;
using NeuralNetwork.Data;

namespace NeuralNetwork.Library
{
    public static class NodeCalculations
    {
        /// <summary>
        ///     Applies the logistic function to an value, returning the result.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double LogisticFunction(double value)
        {
            return 1 / (1 + Math.Pow(Math.E, -value));
        }
    }
}