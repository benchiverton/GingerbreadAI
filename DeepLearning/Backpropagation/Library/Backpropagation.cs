namespace Backpropagation.Library
{
    using NeuralNetwork.Data;
    using NeuralNetwork.Library;

    public class Backpropagation
    {
        public double LearningRate { get; set; }

        public NodeLayerLogic LayerLogic { get; set; }

        public Backpropagation(NodeLayer outputLayer, double learningRate)
        {
            LayerLogic = new NodeLayerLogic
            {
                OutputLayer = outputLayer
            };
            LearningRate = learningRate;
        }

        // should be recursive over the backprop group data
        public void Backpropagate(double[] inputs, double[] targetOutputs)
        {
            var output = LayerLogic.GetResults(inputs);

            // logic for backprop
        }
    }
}