namespace Backpropagation
{
    using NeuralNetwork;
    using NeuralNetwork.Data;

    public class Backpropagation
    {
        public double LearningRate { get; set; }

        public LayerComputor LayerLogic { get; set; }

        public Backpropagation(Layer outputLayer, double learningRate)
        {
            LayerLogic = new LayerComputor
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