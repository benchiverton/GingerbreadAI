using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AI.Test.Framework.Accuracy;
using BackPropagation;
using NeuralNetwork;
using NeuralNetwork.Data;
using NeuralNetwork.Library.Extensions;
using Xunit;

namespace AI.Tests.SineCurve
{
    public class SineCurveUsingBackPropagation
    {
        private const string ResultsDirectory = "SineCurveUsingBackPropagation";

        [RunnableInDebugOnly]
        public void ApproximateSineCurveUsingBackPropagation()
        {
            var input = new Layer("Input", 1, new Layer[0]);
            var inner1 = new Layer("Inner1", 20, new[] { input });
            var inner2 = new Layer("Inner1", 20, new[] { inner1 });
            var outputLayer = new Layer("Output", 1, new[] { inner2 });
            LayerInitialiser.Initialise(new Random(), outputLayer);
            var accuracyResults = new List<double>();
            var initialResults = new double[100];
            var finalResults = new double[100];
            var inputs = new double[100];
            for (var i = 0; i < inputs.Length; i++)
            {
               inputs[i] = (double) i / inputs.Length;
            }
            for (var i = 0; i < inputs.Length; i++)
            {
               initialResults[i] = outputLayer.GetResults(new[] { inputs[i] })[0];
            }

            Parallel.ForEach(new double[4], x => TrainNetwork(outputLayer, inputs, accuracyResults));
            SetResults(inputs, outputLayer, finalResults);

            var suffix = DateTime.Now.Ticks;
            System.IO.Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");
            using (var file = new System.IO.StreamWriter($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{suffix}.csv", false))
            {
               file.WriteLine(string.Join(",", inputs.ToArray()));
               file.WriteLine(string.Join(",", inputs.Select(Calculation)));
               file.WriteLine(string.Join(",", initialResults.ToArray()));
               file.WriteLine(string.Join(",", finalResults.ToArray()));
            }
            using (var file = new System.IO.StreamWriter($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/accuracyResults-{suffix}.csv", false))
            {
               file.WriteLine(string.Join(",", accuracyResults.ToArray()));
            }
        }

        // ignore how inaccurate the accuracy results could be if this is ran in parallel :^)
        private void TrainNetwork(Layer outputLayer, double[] inputs, List<double> accuracyResults)
        {
            var rand = new Random();
            var output = outputLayer.CloneNewWithWeightReferences();
            var backpropagator = new BackPropagator(output, 0.1, LearningRateModifier, 0.9);
            for (var i = 0; i < 100000; i++)
            {
                if(i % 1000 == 0){
                    var currentResults = new double[inputs.Length];
                    SetResults(inputs, output, currentResults);
                    accuracyResults.Add(AccuracyTester.CalculateKolmogorovStatistic(
                        currentResults, inputs.Select(Calculation).ToArray()));
                }
               var trial = rand.NextDouble();
               backpropagator.BackPropagate(new[] { trial }, new double?[] { Calculation(trial) });
            }
        }

        private void SetResults(double[] inputs, Layer output, double[] targetArray)
        {
            for (var i = 0; i < inputs.Length; i++)
            {
               targetArray[i] = output.GetResults(new[] { inputs[i] })[0];
            }
        }

        private static double LearningRateModifier(double rate)
            => rate * 0.99 < 0.1 ? 0.1 : rate * 0.99;

        private static double Calculation(double input)
            => 0.5 * Math.Sin(3 * Math.PI * input) + 0.5;
    }
}