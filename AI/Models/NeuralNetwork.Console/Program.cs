using NeuralNetwork.Extensions;

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
            var inner1 = new Layer("Inner1", 6, new[] { group });
            var inner2 = new Layer("Inner1", 25, new[] { inner1 });
            var inner3 = new Layer("Inner2", 125, new[] { inner1 });
            //var inner3 = new Layer("Inner2", 125, new[] { inner2 });
            var output = new Layer("Output", 1, new[] { inner1 });

            var rand = new Random();
            LayerInitialiser.Initialise(rand, output);
            var copy = output.DeepCopy().SetAllWeightsToZero();

            var nodeLayerLogic = new LayerCalculator
            {
                OutputLayer = output
            };
            Console.WriteLine(output.ToString(true));

            var inputs = new double[100];
            var initialResults = new double[100];
            var finalResults = new double[100];
            var extrapolation = new double[100];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = (double)i / inputs.Length;
            }

            // initial results
            for (var i = 0; i < inputs.Length; i++)
            {
                initialResults[i] = nodeLayerLogic.GetResults(new[] { inputs[i] })[0];
            }

            // perform backpropagation
            var backpropagator = new Backpropagator(output, 0.1, 0.9);
            for (var i = 0; i < 1000000; i++)
            {
                var trial = rand.NextDouble();
                backpropagator.Backpropagate(new[] { trial }, new[] { Calculation(trial) });
            }

            // final results
            for (var i = 0; i < inputs.Length; i++)
            {
                finalResults[i] = nodeLayerLogic.GetResults(new[] { inputs[i] })[0];
            }

            // extrapolation
            for (var i = 0; i < inputs.Length; i++)
            {
                extrapolation[i] = nodeLayerLogic.GetResults(new[] { inputs[i] + 1 })[0];
            }

            using (var file = new System.IO.StreamWriter($@"{Directory.GetCurrentDirectory()}\networkResults.csv", false))
            {
                file.WriteLine(string.Join(",", inputs.ToArray()));
                file.WriteLine(string.Join(",", inputs.Select(Calculation)));
                file.WriteLine(string.Join(",", initialResults.ToArray()));
                file.WriteLine(string.Join(",", finalResults.ToArray()));
                file.WriteLine(string.Join(",", extrapolation.ToArray()));
            }
        }

        private static double Calculation(double input)
        {
            return 0.5 * Math.Sin(6 * Math.PI * input) + 0.5;
        }
    }
}