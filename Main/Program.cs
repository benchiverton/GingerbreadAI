using System;
using NeuralNetwork.Data;
using NeuralNetwork.Library;

namespace Main
{
    public class Program
    {
        public static void Main()
        {
            var n = new NodeNetwork();

            var group = new NodeGroup("Input", 20);
            NodeNetworkCalculations.AddNodeGroup(group, n);

            var inner1 = new NodeGroup("Inner1", 20, new[] {group});
            NodeNetworkCalculations.AddNodeGroup(inner1, n);

            var inner2 = new NodeGroup("Inner2", 20, new[] {group});
            NodeNetworkCalculations.AddNodeGroup(inner2, n);

            var output = new NodeGroup("Output", 20, new[] {inner1, inner2});
            NodeNetworkCalculations.AddNodeGroup(output, n);

            Initialiser.Initialise(new Random(), n);

            Console.WriteLine(n);

            var inputs = new double[20];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = 1;
            }
            var results = NodeNetworkCalculations.GetResult(inputs, n);
            Console.WriteLine($"Results: {string.Join(", ", results)}");
            Console.ReadLine();
        }
    }
}