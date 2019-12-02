using NeuralNetwork;
using NeuralNetwork.Models;

namespace Backpropagation
{
    public class Momentum
    {
        private readonly Layer _momentumDeltaHolder;
        private readonly double _magnitudeOfMomentum;

        public static Momentum GenerateMomentum(Layer outputLayer, double magnitudeOfMomentum)
        {
            return new Momentum(outputLayer.CloneWithNodeReferences(), magnitudeOfMomentum);
        }

        public Momentum StepBackwards(int layerIndex)
        {
            return new Momentum(_momentumDeltaHolder.PreviousLayers[layerIndex], _magnitudeOfMomentum);
        }

        public void ApplyMomentum(Node node, Node prevNode, double change, int nodeIndex)
        {
            var momentumNode = _momentumDeltaHolder.Nodes[nodeIndex];

            node.Weights[prevNode].Value += _magnitudeOfMomentum * momentumNode.Weights[prevNode].Value;
            momentumNode.Weights[prevNode].Value = change + _magnitudeOfMomentum * momentumNode.Weights[prevNode].Value;
        }

        public void ApplyBiasMomentum(Node node, Layer prevLayer, double change, int nodeIndex)
        {
            var momentumNode = _momentumDeltaHolder.Nodes[nodeIndex];

            node.BiasWeights[prevLayer].Value += _magnitudeOfMomentum * momentumNode.BiasWeights[prevLayer].Value;
            momentumNode.BiasWeights[prevLayer].Value = change + _magnitudeOfMomentum * momentumNode.BiasWeights[prevLayer].Value;
        }

        private Momentum(Layer momentumDeltaHolder, double magnitudeOfMomentum)
        {
            _momentumDeltaHolder = momentumDeltaHolder;
            _magnitudeOfMomentum = magnitudeOfMomentum;
        }
    }
}
