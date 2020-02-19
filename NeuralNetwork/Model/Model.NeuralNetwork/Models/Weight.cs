using System;

namespace Model.NeuralNetwork.Models
{
    // This class is a yikes from me
    public class Weight
    {
        public Weight(double value)
        {
            Value = value;
        }

        public virtual double Value { get; set; }
    }
}
