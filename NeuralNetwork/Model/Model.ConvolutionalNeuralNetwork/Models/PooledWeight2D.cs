using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class PooledWeight2D : Weight
    {
        public PooledWeight2D(int dimensions) : base(0)
        {
            _dimensions = dimensions;
            _occurrences = 1;
            CalculateMagnitude();
        }

        private readonly int _dimensions;
        private int _occurrences;
        private double _magnitude;
        private double _value;

        public new double Value { get => _value * _magnitude; set => _value = value; }

        public void IncreaseOccurrences()
        {
            _occurrences++;
            CalculateMagnitude();
        }

        public void CalculateMagnitude()
        {
            _magnitude = (double)_occurrences / (_dimensions * _dimensions);
        }
    }
}
