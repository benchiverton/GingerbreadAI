using System;

namespace NeuralNetwork.Models
{
    // I know, I dislike this code too
    [Serializable]
    public class Weight
    {
        public Weight(double value)
        {
            Value = value;
        }

        public double Value { get; set; }
    }
}
