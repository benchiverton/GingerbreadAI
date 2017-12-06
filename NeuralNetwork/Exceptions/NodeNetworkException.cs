using System;

namespace NeuralNetwork.Exceptions
{
    public class NodeNetworkException : Exception
    {
        public NodeNetworkException() { }
        public NodeNetworkException(string message) : base(message) { }
        public NodeNetworkException(string message, Exception inner) : base(message, inner) { }
    }
}
