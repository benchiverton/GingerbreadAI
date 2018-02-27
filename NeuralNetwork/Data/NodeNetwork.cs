using System.Text;

namespace NeuralNetwork.Data
{
    public class NodeNetwork
    {
        /// <summary>
        ///     An array containing all of the NodeGroups within the network. I feel like the input might not be needed...
        /// </summary>
        public NodeGroup[] Groups;

        public override string ToString()
        {
            var s = new StringBuilder("Your Network:\n");
            foreach (var nodeGroup in Groups)
                s.Append($"{nodeGroup}");
            return s.ToString();
        }
    }
}