namespace NeuralNetwork.Data
{
    using System.Collections.Generic;

    public class NodeWeightReferences
    {
        public NodeLayer CorrespondingNodeLayer;

        public Dictionary<Node, double> CorrespondingNodeWeights;
    }
}
