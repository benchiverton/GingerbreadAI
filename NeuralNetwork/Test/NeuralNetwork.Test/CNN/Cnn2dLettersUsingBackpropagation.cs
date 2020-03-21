using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using DeepLearning.Backpropagation;
using DeepLearning.Backpropagation.Extensions;
using MNIST.IO;
using Model.ConvolutionalNeuralNetwork.Extensions;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork;
using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;
using NeuralNetwork.Test.Helpers;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace NeuralNetwork.Test.CNN
{
    public class Cnn2dLettersUsingBackpropagation
    {
        private const string ResultsDirectory = nameof(Cnn2dLettersUsingBackpropagation);
        private const string TestDataDir = @"C:\Projects\AI\TestData\dogs-vs-cats\test";
        private readonly string _trainingDataDir = $"./{nameof(Cnn2dLettersUsingBackpropagation)}/TrainingData";
        private readonly ITestOutputHelper _testOutputHelper;

        public Cnn2dLettersUsingBackpropagation(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [RunnableInDebugOnly]
        public void TrainAgainstGandWrittenNumbers()
        {
            EnsureDataExists();

            var input = new Layer2D((28, 28), new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.Uniform);
            var filters = (new[] { input }).Add2DConvolutionalLayer(32, (3, 3), ActivationFunctionType.RELU, InitialisationFunctionType.Uniform);
            filters.AddPooling(2);
            var stepDownLayer = new Layer(32, filters.ToArray(), ActivationFunctionType.RELU, InitialisationFunctionType.Uniform);
            var output = new Layer(10, new[] { stepDownLayer }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);
            var momentum = output.GenerateMomentum();
            output.Initialise(new Random());

            foreach (var trainingData in GetDataSet($"{_trainingDataDir}/train-images-idx3-ubyte.gz", $"{_trainingDataDir}/train-labels-idx1-ubyte.gz"))
            {
                var targetOutputs = new double[10];
                targetOutputs[trainingData.label] = 1d;
                output.Backpropagate(trainingData.image, targetOutputs, 0.1, momentum, 0.9);
            }

            var correctResults = new double[10];
            var incorrectResults = new double[10];
            foreach (var trainingData in GetDataSet($"{_trainingDataDir}/t10k-images-idx3-ubyte.gz", $"{_trainingDataDir}/t10k-labels-idx1-ubyte.gz"))
            {
                output.CalculateOutputs(trainingData.image);
                if (output.Nodes[trainingData.label].Output > 0.5)
                {
                    correctResults[trainingData.label]++;
                }
                else
                {
                    incorrectResults[trainingData.label]++;
                }
            }
            _testOutputHelper.WriteLine($"Accuracy detecting 0: {correctResults[0] / (correctResults[0] + incorrectResults[0])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 1: {correctResults[1] / (correctResults[1] + incorrectResults[1])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 2: {correctResults[2] / (correctResults[2] + incorrectResults[2])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 3: {correctResults[3] / (correctResults[3] + incorrectResults[3])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 4: {correctResults[4] / (correctResults[4] + incorrectResults[4])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 5: {correctResults[5] / (correctResults[5] + incorrectResults[5])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 6: {correctResults[6] / (correctResults[6] + incorrectResults[6])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 7: {correctResults[7] / (correctResults[7] + incorrectResults[7])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 8: {correctResults[8] / (correctResults[8] + incorrectResults[8])}");
            _testOutputHelper.WriteLine($"Accuracy detecting 9: {correctResults[9] / (correctResults[9] + incorrectResults[9])}");
        }

        private void EnsureDataExists()
        {
            if (!Directory.Exists(_trainingDataDir))
            {
                Directory.CreateDirectory(_trainingDataDir);
            }

            var directoryFiles = new DirectoryInfo(_trainingDataDir).EnumerateFiles().Select(f => f.Name).ToArray();

            if (!directoryFiles.Contains("train-images-idx3-ubyte.gz"))
            {
                DownloadHelpers.DownloadFile("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", $"{_trainingDataDir}/train-images-idx3-ubyte.gz");
            }
            if (!directoryFiles.Contains("train-labels-idx1-ubyte.gz"))
            {
                DownloadHelpers.DownloadFile("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", $"{_trainingDataDir}/train-labels-idx1-ubyte.gz");
            }
            if (!directoryFiles.Contains("t10k-images-idx3-ubyte.gz"))
            {
                DownloadHelpers.DownloadFile("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", $"{_trainingDataDir}/t10k-images-idx3-ubyte.gz");
            }
            if (!directoryFiles.Contains("t10k-images-idx1-ubyte.gz"))
            {
                DownloadHelpers.DownloadFile("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", $"{_trainingDataDir}/t10k-labels-idx1-ubyte.gz");
            }
        }

        private IEnumerable<(double[] image, int label)> GetDataSet(string imageFileName, string labelFileName)
        {
            var trainingDataSet = FileReaderMNIST.LoadImagesAndLables(labelFileName, imageFileName);

            foreach (var trainingData in trainingDataSet)
            {
                var trainingDataAsDoubleArray = new double[784];
                var trainingDataAsDouble = trainingData.AsDouble();
                for (var i = 0; i < 28; i++)
                {
                    for (var j = 0; j < 28; j++)
                    {
                        trainingDataAsDoubleArray[j + 28 * i] = trainingDataAsDouble[j, i];
                    }
                }
                yield return (trainingDataAsDoubleArray, trainingData.Label);
            }
        }
    }
}
