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

            var layer = new NodeLayer("Input", 20);
            NodeNetworkCalculations.AddNodeLayer(layer, n);

            var inner1 = new NodeLayer("Inner1", 20, new[] {layer});
            NodeNetworkCalculations.AddNodeLayer(inner1, n);

            var inner2 = new NodeLayer("Inner2", 20, new[] {layer});
            NodeNetworkCalculations.AddNodeLayer(inner2, n);

            var output = new NodeLayer("Output", 20, new[] {inner1, inner2});
            NodeNetworkCalculations.AddNodeLayer(output, n);

            Initialiser.Initialise(new Random(), n);

            Console.WriteLine(n);

            var inputs = new double[20];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = -0.001;
            }
            var results = NodeNetworkCalculations.GetResult(inputs, n);
            Console.WriteLine($"Results: {string.Join(", ", results)}");
            Console.ReadLine();
        }
    }
}