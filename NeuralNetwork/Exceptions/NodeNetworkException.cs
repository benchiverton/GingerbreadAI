using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Exceptions
{
    class NodeNetworkException : Exception
    {
        public NodeNetworkException() { }
        public NodeNetworkException(string message) : base(message) { }
        public NodeNetworkException(string message, Exception inner) : base(message, inner) { }
    }
}
