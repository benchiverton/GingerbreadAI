using GingerbreadAI.DeepLearning.Backpropagation.Interfaces;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;

namespace GingerbreadAI.DeepLearning.Backpropagation.Models
{
    public class WeightWithPoolingAndMomentum : WeightWithPooling, IWeightWithMomentum
    {
        internal WeightWithPoolingAndMomentum(WeightWithPooling weightWithPooling) : base(weightWithPooling) => Momentum = 0d;

        public double Momentum { get; set; }
    }
}
