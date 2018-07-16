using NeuralNetwork.Data;

namespace Backpropagation.Data
{
    public class BackpropagationBindingModel
    {
        // we need a reference to the feeding groups in order to peform the backwards pass.
        public NodeLayer NodeGroup { get; set; }
        public BackpropagationBindingModel[] FeedingGroups { get; set; }

        public BackpropagationBindingModel(NodeLayer nodeGroup)
        {
            NodeGroup = nodeGroup;
            FeedingGroups = new BackpropagationBindingModel[0];
        }
    }
}
