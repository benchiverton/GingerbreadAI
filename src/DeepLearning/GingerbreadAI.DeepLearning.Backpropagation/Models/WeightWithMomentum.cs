using GingerbreadAI.DeepLearning.Backpropagation.Interfaces;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.DeepLearning.Backpropagation.Models
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
