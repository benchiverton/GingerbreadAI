namespace Network.Console
{
    using System;
    using System.Linq;
    using NeuralNetwork;
    using NeuralNetwork.Data;

    public class Program
    {
        public static void Main()
        {
            var group = new Layer("Input", 20, new Layer[0]);
            var inner1 = new Layer("Inner1", 20, new[] { group });
            var inner2 = new Layer("Inner2", 20, new[] { group });
            var output = new Layer("Output", 20, new[] { inner1, inner2 });

            var rand = new Random();
            LayerInitialiser.Initialise(rand, output);

            var inputs = new double[20];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = (double)rand.Next(2000000) / 1000000 - 1;
            }
            Console.WriteLine(output.ToString(true));

            var nodeLayerLogic = new LayerComputor
            {
                OutputLayer = output
            };

            var results = nodeLayerLogic.GetResults(inputs);
            Console.WriteLine($"Inputs: {string.Join(", ", inputs.Select(i => Math.Round(i, 3)))}");
            Console.WriteLine($"Results: {string.Join(", ", results.Select(r => Math.Round(r, 3)))}");

            Console.ReadLine();
        }
    }
}