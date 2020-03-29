using System;
using System.Collections.Generic;
using System.Linq;
using System.Timers;
using DeepLearning.Backpropagation;
using DeepLearning.Backpropagation.Extensions;
using Model.ConvolutionalNeuralNetwork.Extensions;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Extensions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;
using Xunit.Abstractions;

namespace NeuralNetwork.Test.Performance
{
    public class Cnn2dPerformanceTest
    {
        private readonly ITestOutputHelper _testOutputHelper;

        private const int IntervalInMs = 5000;
        private int _processedImages;

        public Cnn2dPerformanceTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [RunnableInDebugOnly]
        public void PerformanceTestCnnNetwork()
        {
            var input = new Layer2D((10, 10), new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var filters = new[] { input }.Add2DConvolutionalLayer(12, (4, 4), ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            filters.AddPooling((2, 2));
            var stepDownLayer = new Layer(30, filters.ToArray(), ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(3, new[] { stepDownLayer }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.GlorotUniform);
            output.AddMomentumRecursively();
            output.Initialise(new Random());

            var timer = new Timer
            {
                Interval = IntervalInMs
            };
            timer.Elapsed += OnTimerElapsed;
            timer.Start();
            for (var i = 0; i < 10000; i++)
            {
                output.Backpropagate(SquareAsArray, new [] {1d, 0d, 0d}, 0.1, 0.9);
                _processedImages++;
                output.Backpropagate(SquareAsArray, new [] {1d, 0d, 0d}, 0.1, 0.9);
                _processedImages++;
                output.Backpropagate(SquareAsArray, new [] {1d, 0d, 0d}, 0.1, 0.9);
                _processedImages++;
            }
        }

        private void OnTimerElapsed(object source, ElapsedEventArgs e)
        {
            _testOutputHelper.WriteLine($"Images processed in {IntervalInMs}ms: {_processedImages}");
            _processedImages = 0;
        }

        private double[] SquareAsArray => TransformTo1dArray(new double[,]
        {
            { 0,0,0,0,0,0,0,0,0,0 },
            { 0,1,1,1,1,1,1,1,1,0 },
            { 0,1,1,1,1,1,1,1,1,0 },
            { 0,1,1,0,0,0,0,1,1,0 },
            { 0,1,1,0,0,0,0,1,1,0 },
            { 0,1,1,0,0,0,0,1,1,0 },
            { 0,1,1,0,0,0,0,1,1,0 },
            { 0,1,1,1,1,1,1,1,1,0 },
            { 0,1,1,1,1,1,1,1,1,0 },
            { 0,0,0,0,0,0,0,0,0,0 },
        }).ToArray();

        private double[] CircleAsArray => TransformTo1dArray(new double[,]
        {
            { 0,0,0,0,0,0,0,0,0,0 },
            { 0,0,0,0,1,1,0,0,0,0 },
            { 0,0,1,1,1,1,1,1,0,0 },
            { 0,0,1,1,0,0,1,1,0,0 },
            { 0,1,1,0,0,0,0,1,1,0 },
            { 0,1,1,0,0,0,0,1,1,0 },
            { 0,0,1,1,0,0,1,1,0,0 },
            { 0,0,1,1,1,1,1,1,0,0 },
            { 0,0,0,0,1,1,0,0,0,0 },
            { 0,0,0,0,0,0,0,0,0,0 },
        }).ToArray();

        private double[] TriangleAsArray => TransformTo1dArray(new double[,]
        {
            { 0,0,0,0,0,0,0,0,0,0 },
            { 0,0,0,0,1,1,0,0,0,0 },
            { 0,0,0,1,1,1,1,0,0,0 },
            { 0,0,1,1,0,0,1,1,0,0 },
            { 0,1,1,0,0,0,0,1,1,0 },
            { 1,1,0,0,0,0,0,0,1,1 },
            { 1,0,0,0,0,0,0,0,0,1 },
            { 1,1,1,1,1,1,1,1,1,1 },
            { 1,1,1,1,1,1,1,1,1,1 },
            { 0,0,0,0,0,0,0,0,0,0 },
        }).ToArray();

        private static IEnumerable<double> TransformTo1dArray(double[,] inputData)
        {
            for (var i = 0; i < 10; i++)
            {
                for (var j = 0; j < 10; j++)
                {
                    yield return inputData[i, j];
                }
            }
        }
    }
}
