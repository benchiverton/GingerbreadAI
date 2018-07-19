namespace Backpropagation
{
    using System.Linq;
    using Bens.WonderfulLibrary.Calculations;
    using Bens.WonderfulLibrary.Extensions;
    using NeuralNetwork;
    using NeuralNetwork.Data;

    public class Backpropagation
    {
        public double LearningRate { get; set; }

        public LayerComputor LayerComputor { get; set; }

        public Backpropagation(Layer outputLayer, double learningRate)
        {
            LayerComputor = new LayerComputor
            {
                OutputLayer = outputLayer
            };
            LearningRate = learningRate;
        }

        public void Backpropagate(double[] inputs, double[] targetOutputs)
        {
            var currentLayer = LayerComputor.OutputLayer;
            var curretOutputs = LayerComputor.GetResults(inputs);

            // initial calculations for output layer
            currentLayer.Nodes.Each((node, i) =>
            {
                var delta = BackpropagationCalculations.GetDeltaOutput(curretOutputs[i], targetOutputs[i]);
                foreach (var prevNode in node.Weights.Keys.ToList())
                {
                    node.Weights[prevNode] = node.Weights[prevNode] - (LearningRate * delta * prevNode.Output);
                }
            });
            //RecurseBackpropagation()
        }

        //private void RecurseBackpropagation(Layer layer)
        //{
        //    if (layer.PreviousLayers.Length == 0)
        //    {
        //        return;
        //    }

        //    foreach (var prevLayer in layer.PreviousLayers)
        //    {
        //        RecurseBackpropagation(prevLayer);
        //    }

        //}
    }
}