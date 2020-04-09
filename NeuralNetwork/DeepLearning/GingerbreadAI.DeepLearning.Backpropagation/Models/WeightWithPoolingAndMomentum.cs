using DeepLearning.Backpropagation.Interfaces;
using Model.ConvolutionalNeuralNetwork.Models;

namespace DeepLearning.Backpropagation.Models
{
    public class WeightWithPoolingAndMomentum : WeightWithPooling, IWeightWithMomentum
    {
        internal WeightWithPoolingAndMomentum(WeightWithPooling weightWithPooling) : base(weightWithPooling)
        {
            Momentum = 0d;
        }

        public double Momentum { get; set; }
    }
}
