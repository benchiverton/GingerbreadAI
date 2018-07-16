using Backpropagation.Data;

namespace Backpropagation.Library
{
    using NeuralNetwork.Data;
    using NeuralNetwork.Library;

    public class Backpropagation
    {
        public double LearningRate { get; set; }

        public BackpropagationBindingModel BindingModel { get; set; }

        public NodeLayerLogic LayerLogic { get; set; }

        public Backpropagation(NodeLayer outputLayer, double learningRate)
        {
            BindingModel = BackpropagationMethods.GenerateBackpropagationBindingModel(outputLayer);
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

        //// probably needs to return the deltas ??
        //private void BackPropogate(BackpropagationBindingModel bindingModel)
        //{
        //    if (bindingModel.FeedingGroups.Length == 0)
        //    {
        //        bindingModel.BoundNodeLayer.PreviousGroups.Each((prevGroup, i) =>
        //        {

        //        }
        //        return;
        //    }
        //}
    }
}