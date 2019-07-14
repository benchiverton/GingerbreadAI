using System;

namespace NeuralNetwork.Models
{
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
