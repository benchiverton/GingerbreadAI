using System;
using NeuralNetwork.Data;
using NeuralNetwork.Library;

namespace Main
{
    using System.Linq;

    public class Program
    {
        public static void Main()
        {
            var group = new NodeLayer("Input", 20);
            var inner1 = new NodeLayer("Inner1", 20, new[] { group });
            var inner2 = new NodeLayer("Inner2", 20, new[] { group });
            var output = new NodeLayer("Output", 20, new[] { inner1, inner2 });

            var rand = new Random();
            Initialiser.Initialise(rand, output);

            var inputs = new double[20];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = (double)rand.Next(2000000) / 1000000 - 1;
            }
            Console.WriteLine(output.ToString(true));

            var nodeLayerLogic = new NodeLayerLogic
            {
                OutputLayer = output
            };
            nodeLayerLogic.PopulateResults(inputs);

            var results = output.Outputs;
            Console.WriteLine($"Inputs: {string.Join(", ", inputs.Select(i => Math.Round(i, 3)))}");
            Console.WriteLine($"Results: {string.Join(", ", results.Select(r => Math.Round(r, 3)))}");

            Console.ReadLine();
        }
    }
}