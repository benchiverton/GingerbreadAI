namespace Network.Console
{
    using System;
    using System.IO;
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
            var inner2 = new Layer("Inner2", 100, new[] { inner1 });
            var output = new Layer("Output", 1, new[] { inner2 });

            var rand = new Random();
            LayerInitialiser.Initialise(rand, output);
            var nodeLayerLogic = new LayerComputor
            {
                OutputLayer = output
            };
            Console.WriteLine(output.ToString(true));

            var inputs = new double[100];
            var initialResults = new double[100];
            var finalResults = new double[100];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = (double)i / inputs.Length;
            }

            // initial results
            for (var i = 0; i < inputs.Length; i++)
            {
                initialResults[i] = nodeLayerLogic.GetResults(new[] { inputs[i] })[0];
            }

            // perform backprop
            var backprop = new Backpropagation(output, 0.5);
            for (var i = 0; i < 100000; i++)
            {
                var trial = rand.NextDouble();
                backprop.Backpropagate(new[] { trial }, new[] { 0.5 * Math.Sin(2 * Math.PI * trial) + 0.5});
            }

            // final results
            for (var i = 0; i < inputs.Length; i++)
            {
                finalResults[i] = nodeLayerLogic.GetResults(new[] { inputs[i] })[0];
            }

            using (var file = new System.IO.StreamWriter($@"{Directory.GetCurrentDirectory()}\networkResults.csv", false))
            {
                file.WriteLine(string.Join(",", inputs.ToArray()));
                file.WriteLine(string.Join(",", initialResults.ToArray()));
                file.WriteLine(string.Join(",", finalResults.ToArray()));
            }
        }
    }
}