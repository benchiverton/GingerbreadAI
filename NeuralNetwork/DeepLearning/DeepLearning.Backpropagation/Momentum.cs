using Model.NeuralNetwork;
using Model.NeuralNetwork.Models;

namespace DeepLearning.Backpropagation
{
    public class Momentum
    {
        private readonly Layer _momentumDeltaHolder;
        private readonly double _magnitudeOfMomentum;

        public static Momentum GenerateMomentum(Layer outputLayer, double magnitudeOfMomentum)
        {
            return new Momentum(outputLayer.CloneWithSameWeightKeyReferences(), magnitudeOfMomentum);
        }

        public Momentum StepBackwards(int layerIndex)
        {
            return new Momentum(_momentumDeltaHolder.PreviousLayers[layerIndex], _magnitudeOfMomentum);
        }

        public void ApplyMomentum(Node prevNode, Weight prevNodeWeight, double change, int nodeIndex)
        {
            var momentumNode = _momentumDeltaHolder.Nodes[nodeIndex];
            var momentumWeight = momentumNode.Weights[prevNode];

            prevNodeWeight.Value += _magnitudeOfMomentum * momentumWeight.Value;
            momentumWeight.Value = change + _magnitudeOfMomentum * momentumWeight.Value;
        }

        public void ApplyBiasMomentum(Layer prevLayer, Weight prevLayerWeight, double change, int nodeIndex)
        {
            var momentumNode = _momentumDeltaHolder.Nodes[nodeIndex];
            var momentumWeight = momentumNode.BiasWeights[prevLayer];

            prevLayerWeight.Value += _magnitudeOfMomentum * momentumWeight.Value;
            momentumWeight.Value = change + _magnitudeOfMomentum * momentumWeight.Value;
        }

        private Momentum(Layer momentumDeltaHolder, double magnitudeOfMomentum)
        {
            _momentumDeltaHolder = momentumDeltaHolder;
            _magnitudeOfMomentum = magnitudeOfMomentum;
        }
    }
}
