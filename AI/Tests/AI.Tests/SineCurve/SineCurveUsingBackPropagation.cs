using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Backpropagation;
using NeuralNetwork;

namespace AI.Tests.SineCurve
{
    using Calculations.Statistics;
    using NeuralNetwork.Models;
    using Xunit.Abstractions;

    public class SineCurveUsingBackpropagation
    {
        private const string ResultsDirectory = nameof(SineCurveUsingBackpropagation);
        private readonly ITestOutputHelper _testOutputHelper;

        public SineCurveUsingBackpropagation(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [RunnableInDebugOnly]
        public void ApproximateSineCurveUsingBackpropagation()
        {
            var input = new Layer("Input", 1, new Layer[0]);
            var inner1 = new Layer("Inner1", 5, new[] { input });
            var inner2 = new Layer("Inner2", 25, new[] { inner1 });
            var inner3 = new Layer("Inner3", 5, new[] { inner2 });
            var outputLayer = new Layer("Output", 1, new[] { inner3 });
            _testOutputHelper.WriteLine(outputLayer.ToString(true));
            outputLayer.Initialise(new Random());
            var accuracyResults = new List<double>();
            var initialResults = new double[100];
            var finalResults = new double[100];
            var inputs = new double[100];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = (double)i / inputs.Length;
            }
            for (var i = 0; i < inputs.Length; i++)
            {
                initialResults[i] = outputLayer.GetResults(new[] { inputs[i] })[0];
            }

            Parallel.For(0, 4, x => TrainNetwork(outputLayer, inputs, accuracyResults));
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
            var output = outputLayer.CloneWithNodeAndWeightReferences();
            var momentum = Momentum.GenerateMomentum(output, 0.9);
            var learningRate = 0.25;
            for (var i = 0; i < 100000; i++)
            {
                if (i % 1000 == 0)
                {
                    var currentResults = new double[inputs.Length];
                    SetResults(inputs, output, currentResults);
                    accuracyResults.Add(AccuracyStatistics.CalculateKolmogorovStatistic(
                        currentResults, inputs.Select(Calculation).ToArray()));
                }
                var trial = rand.NextDouble();
                output.Backpropagate(new[] { trial }, new double[] { Calculation(trial) }, learningRate, momentum);
                ModifyLearningRate(ref learningRate);
            }
        }

        private void SetResults(double[] inputs, Layer output, double[] targetArray)
        {
            for (var i = 0; i < inputs.Length; i++)
            {
                targetArray[i] = output.GetResults(new[] { inputs[i] })[0];
            }
        }

        private static void ModifyLearningRate(ref double rate)
        {
            rate = rate * 0.99 < 0.1 ? 0.1 : rate * 0.99;
        }

        private static double Calculation(double input)
            => 0.5 * Math.Sin(3 * Math.PI * input) + 0.5;
    }
}