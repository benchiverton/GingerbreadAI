using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DeepLearning.Backpropagation;
using DeepLearning.Backpropagation.Extensions;
using Library.Computations.Statistics;
using Model.NeuralNetwork;
using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;
using Xunit.Abstractions;

namespace NeuralNetwork.Test.NN
{
    public class SineCurveUsingBackPropagation_old
    {
        private const string ResultsDirectory = nameof(SineCurveUsingBackPropagation_old);
        private readonly ITestOutputHelper _testOutputHelper;

        public SineCurveUsingBackPropagation_old(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [RunnableInDebugOnly]
        public void ApproximateSineCurveUsingBackpropagation()
        {
            var input = new Layer(1, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var inner1 = new Layer(10, new[] { input }, ActivationFunctionType.Tanh, InitialisationFunctionType.HeEtAl);
            var inner2 = new Layer(10, new[] { inner1 }, ActivationFunctionType.Tanh, InitialisationFunctionType.HeEtAl);
            var outputLayer = new Layer(1, new[] { inner2 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.None);
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

            outputLayer.Initialise(new Random());
            var rand = new Random();
            var momentum = outputLayer.GenerateMomentum();
            for (var i = 0; i < 100000; i++)
            {
                if (i % 1000 == 0)
                {
                    var currentResults = new double[inputs.Length];
                    SetResults(inputs, outputLayer, currentResults);
                    accuracyResults.Add(AccuracyStatistics.CalculateKolmogorovStatistic(
                        currentResults, inputs.Select(Calculation).ToArray()));
                }
                var trial = rand.NextDouble();
                outputLayer.Backpropagate(new[] { trial }, new double[] { Calculation(trial) }, 0.1, momentum, 0.9);
            }

            SetResults(inputs, outputLayer, finalResults);

            var suffix = DateTime.Now.Ticks;
            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");
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

        private void SetResults(double[] inputs, Layer output, double[] targetArray)
        {
            for (var i = 0; i < inputs.Length; i++)
            {
                targetArray[i] = output.GetResults(new[] { inputs[i] })[0];
            }
        }

        private static double Calculation(double input) => 0.5 * Math.Sin(3 * Math.PI * input) + 0.5;
    }
}