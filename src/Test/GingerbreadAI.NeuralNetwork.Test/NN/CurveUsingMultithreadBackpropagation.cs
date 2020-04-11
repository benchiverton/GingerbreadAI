﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GingerbreadAI.DeepLearning.Backpropagation;
using GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions;
using GingerbreadAI.DeepLearning.Backpropagation.Extensions;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Extensions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using GingerbreadAI.NeuralNetwork.Test.Statistics;
using Xunit.Abstractions;

namespace GingerbreadAI.NeuralNetwork.Test.NN
{
    public class CurveUsingMultiThreadBackpropagation
    {
        private const string ResultsDirectory = nameof(CurveUsingMultiThreadBackpropagation);
        private readonly ITestOutputHelper _testOutputHelper;

        public CurveUsingMultiThreadBackpropagation(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [RunnableInDebugOnly]
        public void ApproximateCurveUsingMultipleThreads()
        {
            var input = new Layer(1, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var inner = new Layer(20, new[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.HeEtAl);
            var outputLayer = new Layer(1, new[] { inner }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.None);
            outputLayer.AddMomentumRecursively();
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

            var threadCount = 4;
            var currentThread = 0;
            Parallel.For(0, threadCount, x => TrainNetwork(outputLayer, inputs, accuracyResults, threadCount, currentThread++));
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

        private void TrainNetwork(Layer outputLayer, double[] inputs, List<double> accuracyResults, int threadCount, int currentThread)
        {
            var rand = new Random();
            var output = outputLayer.CloneWithSameWeightValueReferences();

            for (var i = 0; i < 10000; i++)
            {
                if (i % 100 == 0)
                {
                    var currentResults = new double[inputs.Length];
                    SetResults(inputs, output, currentResults);
                    accuracyResults.Add(AccuracyStatistics.CalculateKolmogorovStatistic(
                        currentResults, inputs.Select(Calculation).ToArray()));
                }
                var trial = rand.NextDouble() / 4 + ((double)currentThread + 1) / threadCount;
                output.Backpropagate(new[] { trial }, new [] { Calculation(trial) }, ErrorFunctionType.MSE, 0.01, 0.9);
            }
        }

        private void SetResults(double[] inputs, Layer output, double[] targetArray)
        {
            for (var i = 0; i < inputs.Length; i++)
            {
                targetArray[i] = output.GetResults(new[] { inputs[i] })[0];
            }
        }

        private static double Calculation(double input) => input * input;
    }
}
