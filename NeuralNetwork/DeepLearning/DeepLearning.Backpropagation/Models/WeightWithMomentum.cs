using DeepLearning.Backpropagation.Interfaces;
using Model.NeuralNetwork.Models;

namespace DeepLearning.Backpropagation.Models
{
    public class WeightWithMomentum : Weight, IWeightWithMomentum
    {
        public WeightWithMomentum(double value) : base(value)
        {
            Momentum = 0d;
        }

        public double Momentum { get; set; }
    }
}
