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
            var inner1 = new Layer("Inner1", 10, new[] { group });
            var inner2 = new Layer("Inner2", 20, new[] { inner1 });
            var inner3 = new Layer("Inner2", 20, new[] { inner2 });
            var inner4 = new Layer("Inner2", 10, new[] { inner3 });
            var output = new Layer("Output", 1, new[] { inner4 });

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
            var backprop = new Backpropagation(output, 0.3);
            for (var i = 0; i < 1000000; i++)
            {
                var trial = rand.NextDouble();
                backprop.Backpropagate(new[] { trial }, new[] { Math.Sin(trial * Math.PI) });
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