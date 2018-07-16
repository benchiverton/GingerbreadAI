using NeuralNetwork.Data;

namespace Backpropagation.Data
{
    public class BackpropagationBindingModel
    {
        // we need a reference to the feeding groups in order to peform the backwards pass.
        public NodeLayer BoundNodeLayer { get; set; }
        public BackpropagationBindingModel[] FeedingGroups { get; set; }

        public BackpropagationBindingModel(NodeLayer nodeGroup)
        {
            BoundNodeLayer = nodeGroup;
            FeedingGroups = new BackpropagationBindingModel[0];
        }
    }
}
