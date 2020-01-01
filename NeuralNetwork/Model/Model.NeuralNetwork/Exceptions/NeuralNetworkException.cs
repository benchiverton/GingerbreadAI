using System;

namespace Model.NeuralNetwork.Exceptions
{
    public class NeuralNetworkException : Exception
    {
        public NeuralNetworkException()
        {
        }

        public NeuralNetworkException(string message) : base(message)
        {
        }

        public NeuralNetworkException(string message, Exception inner) : base(message, inner)
        {
        }
    }
}