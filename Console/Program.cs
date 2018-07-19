namespace Network.Console
{
    using System;
    using System.Linq;
    using Backpropagation;
    using NeuralNetwork;
    using NeuralNetwork.Data;

    public class Program
    {
        public static void Main()
        {
            var group = new Layer("Input", 1, new Layer[0]);
            var inner1 = new Layer("Inner1", 100, new[] { group });
            var inner2 = new Layer("Inner2", 100, new[] { group });
            var output = new Layer("Output", 1, new[] { inner1 });

            var rand = new Random();
            LayerInitialiser.Initialise(rand, output);
            var nodeLayerLogic = new LayerComputor
            {
                OutputLayer = output
            };
            Console.WriteLine(output.ToString(true));

            var inputs = new double[1000];
            var initialResults = new double[1000];
            var finalResults = new double[1000];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = (double)rand.Next(1000000) / 1000000 * 3.14;
            }
            for (var i = 0; i < inputs.Length; i++)
            {
                initialResults[i] = nodeLayerLogic.GetResults(new[] { inputs[i] })[0];
            }

            var backprop = new Backpropagation(output, 0.5);
            for (var i = 0; i < 10000; i++)
            {
                var trial = (double)rand.Next(1000000) / 1000000 * 3.14;
                backprop.Backpropagate(new[] { trial }, new[] { Math.Sin(trial) });
            }

            for (var i = 0; i < inputs.Length; i++)
            {
                finalResults[i] = nodeLayerLogic.GetResults(new[] { inputs[i] })[0];
            }

            Console.WriteLine($"Inputs: {string.Join(",", inputs.Select(i => Math.Round(i, 3)))}");
            Console.WriteLine($"Results: {string.Join(",", initialResults.Select(r => Math.Round(r, 3)))}");
            Console.WriteLine($"Results: {string.Join(",", finalResults.Select(r => Math.Round(r, 3)))}");

            Console.ReadLine();
        }
    }
}