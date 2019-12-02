using NeuralNetwork;
using NeuralNetwork.Models;
using System;
using Xunit.Abstractions;

namespace AI.Tests.MultipleInputs
{
    using System.Collections.Generic;
    using System.IO;
    using Backpropagation;

    public class MultipleInputsUsingBackpropagation
    {
        private const string ResultsDirectory = nameof(MultipleInputsUsingBackpropagation);
        private readonly ITestOutputHelper _testOutputHelper;

        public MultipleInputsUsingBackpropagation(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [RunnableInDebugOnly]
        public void PredictResultsFromMultipleInputs()
        {
            var input1 = new Layer("input1", 1, new Layer[0]);
            var input2 = new Layer("input2", 1, new Layer[0]);
            var inner = new Layer("Inner", 20, new[] { input1, input2 });
            var outputLayer = new Layer("Output", 1, new[] { inner });
            outputLayer.Initialise(new Random());
            _testOutputHelper.WriteLine(outputLayer.ToString(true));

            var actualResults = new double[6, 6];
            for (var i = 0; i < 6; i++)
            {
                for (var j = 0; j < 6; j++)
                {
                    actualResults[i, j] = Calculation((double)i / 5, (double)j / 5);
                }
            }
            var initialResults = new double[6, 6];
            for (var i = 0; i < 6; i++)
            {
                for (var j = 0; j < 6; j++)
                {
                    initialResults[i, j] = outputLayer.GetResults(new Dictionary<Layer, double[]>
                    {
                        {input1, new[] {(double) i / 5}},
                        {input2, new[] {(double) j / 5}}
                    })[0];
                }
            }

            var inputDict = new Dictionary<Layer, double[]>()
            {
                {input1, new[] { (double)0 }},
                {input2, new[] { (double)0 }}
            };

            var learningRate = 0.1;
            var rand = new Random();
            for (var i = 0; i < 100000; i++)
            {
                var inputValue1 = rand.NextDouble();
                var inputValue2 = rand.NextDouble();
                inputDict[input1][0] = inputValue1;
                inputDict[input2][0] = inputValue2;
                outputLayer.Backpropagate(inputDict, new double?[] { Calculation(inputValue1, inputValue2) }, learningRate);

                ModifyLearningRate(ref learningRate);
            }

            var finalResults = new double[6, 6];
            for (var i = 0; i < 6; i++)
            {
                for (var j = 0; j < 6; j++)
                {
                    finalResults[i, j] = outputLayer.GetResults(new Dictionary<Layer, double[]>
                    {
                        {input1, new[] {(double) i / 5}},
                        {input2, new[] {(double) j / 5}}
                    })[0];
                }
            }

            var suffix = DateTime.Now.Ticks;
            System.IO.Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");
            using (var file = new System.IO.StreamWriter($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{suffix}.csv", false))
            {
                WriteResultToFile(file, actualResults);
                WriteResultToFile(file, initialResults);
                WriteResultToFile(file, finalResults);
            }
        }

        private void WriteResultToFile(StreamWriter file, double[,] values)
        {
            for (var i = 0; i < 6; i++)
            {
                for (var j = 0; j < 6; j++)
                {
                    file.Write($"{values[i, j]},");
                }
                file.WriteLine();
            }
        }

        private static void ModifyLearningRate(ref double rate)
        {
            rate = rate * 0.99 < 0.1 ? 0.1 : rate * 0.99;
        }

        private static double Calculation(double input1, double input2)
            => (input1 + input2) / 2;
    }
}
