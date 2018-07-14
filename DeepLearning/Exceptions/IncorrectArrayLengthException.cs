using System;

namespace DeepLearning.Exceptions
{
    public class IncorrectArrayLengthException : Exception
    {
        public IncorrectArrayLengthException()
        {
        }

        public IncorrectArrayLengthException(string message) : base(message)
        {
        }

        public IncorrectArrayLengthException(string message, Exception inner) : base(message, inner)
        {
        }
    }
}