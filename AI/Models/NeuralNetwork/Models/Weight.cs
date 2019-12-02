using System;

namespace NeuralNetwork.Models
{
    // This class is a yikes from me
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
