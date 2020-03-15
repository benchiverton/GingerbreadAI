using System;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class PooledWeight : Weight
    {
        internal PooledWeight(int weightDimension, int poolSize) : base(0)
        {
            _weightDimension = weightDimension;
            _poolSize = poolSize;
            _occurrences = 1;
            CalculateMagnitude();
        }

        private readonly int _weightDimension;
        private readonly int _poolSize;
        private int _occurrences;
        private double _magnitude;
        private double _value;

        public override double Value { get => _value * _magnitude; set => _value = value; }

        public void IncreaseOccurrences()
        {
            _occurrences++;
            CalculateMagnitude();
        }

        public void CalculateMagnitude()
        {
            _magnitude = (double)_occurrences / Math.Pow(_poolSize, _weightDimension);
        }
    }
}
