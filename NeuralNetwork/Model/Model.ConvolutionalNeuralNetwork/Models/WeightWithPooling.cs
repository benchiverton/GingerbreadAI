using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class WeightWithPooling : Weight
    {
        private readonly int _poolSize;
        private int _occurrences;
        private double _magnitude;
        private double _value;

        internal WeightWithPooling(int poolSize, double value) : base(value)
        {
            _poolSize = poolSize;
            _occurrences = 1;
            CalculateMagnitude();
        }

        protected WeightWithPooling(WeightWithPooling weightWithPooling) : base(weightWithPooling._value)
        {
            _poolSize = weightWithPooling._poolSize;
            _occurrences = weightWithPooling._occurrences;
            CalculateMagnitude();
        }

        public override double Value { get => _value * _magnitude; set => _value = value; }

        public void IncreaseOccurrences()
        {
            _occurrences++;
            CalculateMagnitude();
        }

        public void CalculateMagnitude()
        {
            _magnitude = (double)_occurrences / _poolSize;
        }
    }
}
