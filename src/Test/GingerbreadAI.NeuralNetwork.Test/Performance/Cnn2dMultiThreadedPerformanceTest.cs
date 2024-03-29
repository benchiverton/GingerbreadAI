using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Timers;
using GingerbreadAI.DeepLearning.Backpropagation;
using GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions;
using GingerbreadAI.DeepLearning.Backpropagation.Extensions;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Extensions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit.Abstractions;

namespace GingerbreadAI.NeuralNetwork.Test.Performance;

public class Cnn2dMultiThreadedPerformanceTest
{
    private readonly ITestOutputHelper _testOutputHelper;

    private const int IntervalInMs = 5000;
    private const int TotalSamples = 5;
    private int _sampleCount;
    private int _processedImages;
    private bool _continueProcessing = true;

    public Cnn2dMultiThreadedPerformanceTest(ITestOutputHelper testOutputHelper) => _testOutputHelper = testOutputHelper;

    [RunnableInDebugOnly]
    public void PerformanceMultiThreadedTestCnnNetwork()
    {
        var input = new Layer2D((10, 10), Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.None);
        var filters = new[] { input }.Add2DConvolutionalLayer(16, (3, 3), ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
        filters.AddMaxPooling((2, 2));
        var stepDownLayer = new Layer(30, filters.ToArray(), ActivationFunctionType.Tanh, InitialisationFunctionType.HeUniform);
        var output = new Layer(3, new[] { stepDownLayer }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.GlorotUniform);
        output.AddMomentumRecursively();
        output.Initialise(new Random());

        _testOutputHelper.WriteLine($"Starting test run: Interval: {IntervalInMs}ms, Samples: {TotalSamples}");

        var timer = new Timer
        {
            Interval = IntervalInMs
        };
        timer.Elapsed += OnTimerElapsed;
        timer.Start();

        Parallel.For(0, 4, i =>
        {
            var networkToTrainWith = output.CloneWithDifferentOutputs();
            while (_continueProcessing)
            {
                networkToTrainWith.Backpropagate(SquareAsArray, new[] { 1d, 0d, 0d }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                _processedImages++;
                networkToTrainWith.Backpropagate(CircleAsArray, new[] { 0d, 1d, 0d }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                _processedImages++;
                networkToTrainWith.Backpropagate(TriangleAsArray, new[] { 0d, 0d, 1d }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                _processedImages++;
            }
        });
        timer.Dispose();

        output.CalculateOutputs(SquareAsArray);
        _testOutputHelper.WriteLine($"Results after training from Square: Square: {output.Nodes[0].Output:0.000}; Circle: {output.Nodes[1].Output:0.000}, Triangle: {output.Nodes[2].Output:0.000}");
        output.CalculateOutputs(CircleAsArray);
        _testOutputHelper.WriteLine($"Results after training from Circle: Square: {output.Nodes[0].Output:0.000}; Circle: {output.Nodes[1].Output:0.000}, Triangle: {output.Nodes[2].Output:0.000}");
        output.CalculateOutputs(TriangleAsArray);
        _testOutputHelper.WriteLine($"Results after training from Triangle: Square: {output.Nodes[0].Output:0.000}; Circle: {output.Nodes[1].Output:0.000}, Triangle: {output.Nodes[2].Output:0.000}");
    }

    private void OnTimerElapsed(object source, ElapsedEventArgs e)
    {
        _testOutputHelper.WriteLine($"Images processed in {IntervalInMs}ms: {_processedImages}");
        _sampleCount++;
        if (_sampleCount < TotalSamples)
        {
            _processedImages = 0;
        }
        else
        {
            _continueProcessing = false;
        }
    }

    private static double[] SquareAsArray => TransformTo1dArray(new double[,]
    {
            { 0,0,0,0,0,0,0,0,0,0 },
            { 0,1,1,1,1,1,1,1,1,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 0,1,1,1,1,1,1,1,1,0 },
            { 0,0,0,0,0,0,0,0,0,0 },
    }).ToArray();

    private static double[] CircleAsArray => TransformTo1dArray(new double[,]
    {
            { 0,0,0,0,0,0,0,0,0,0 },
            { 0,0,0,0,1,1,0,0,0,0 },
            { 0,0,1,1,0,0,1,1,0,0 },
            { 0,0,1,0,0,0,0,1,0,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 0,0,1,0,0,0,0,1,0,0 },
            { 0,0,1,1,0,0,1,1,0,0 },
            { 0,0,0,0,1,1,0,0,0,0 },
            { 0,0,0,0,0,0,0,0,0,0 },
    }).ToArray();

    private static double[] TriangleAsArray => TransformTo1dArray(new double[,]
    {
            { 0,0,0,0,0,0,0,0,0,0 },
            { 0,0,0,0,1,1,0,0,0,0 },
            { 0,0,0,0,1,1,0,0,0,0 },
            { 0,0,0,1,0,0,1,0,0,0 },
            { 0,0,1,0,0,0,0,1,0,0 },
            { 0,1,0,0,0,0,0,0,1,0 },
            { 1,0,0,0,0,0,0,0,0,1 },
            { 1,1,1,1,1,1,1,1,1,1 },
            { 0,0,0,0,0,0,0,0,0,0 },
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
