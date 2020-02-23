using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using DeepLearning.Backpropagation;
using Model.ConvolutionalNeuralNetwork.Extensions;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork;
using Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;

namespace NeuralNetwork.Test.CNN
{
    public class CnnNetworkUsingBackpropagation
    {
        private const string ResultsDirectory = nameof(CnnNetworkUsingBackpropagation);
        private const string TrainingDataDir = @"C:\Projects\AI\TestData\dogs-vs-cats\train";
        private const string TestDataDir = @"C:\Projects\AI\TestData\dogs-vs-cats\test";
        private readonly ITestOutputHelper _testOutputHelper;

        public CnnNetworkUsingBackpropagation(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [RunnableInDebugOnly]
        public void PredictCatVsDog()
        {
            var inputR = new Layer2D((101, 101), new Layer[0]);
            var inputG = new Layer2D((101, 101), new Layer[0]);
            var inputB = new Layer2D((101, 101), new Layer[0]);
            var filters = (new[] { inputR, inputG, inputB }).Add2DConvolutionalLayer(32, 3);
            filters.AddPooling(3);
            var stepDownLayer = new Layer(32, filters.ToArray());
            stepDownLayer.Initialise(new Random());
            var output = new Layer(2, new[] { stepDownLayer });
            var momentum = Momentum.GenerateMomentum(output, 0.9);

            var trainingDataCat = GetImageData("cat", TrainingDataDir, inputR, inputG, inputB).GetEnumerator();
            var trainingDataDog = GetImageData("dog", TrainingDataDir, inputR, inputG, inputB).GetEnumerator();
            var i = 0;
            while (trainingDataCat.MoveNext() && trainingDataDog.MoveNext() && i < 1000)
            {
                output.Backpropagate(trainingDataCat.Current, new[] { 1d, 0d }, 0.1, momentum);
                output.Backpropagate(trainingDataDog.Current, new[] { 0d, 1d }, 0.1, momentum);
                i++;
            }

            var correctCatResults = 0;
            var correctDogResults = 0;
            var incorrectCatResults = 0;
            var incorrectDogResults = 0;
            foreach (var testCatData in GetImageData("cat", TestDataDir, inputR, inputG, inputB))
            {
                output.PopulateAllOutputs(testCatData);
                if (output.Nodes[0].Output > output.Nodes[1].Output)
                {
                    correctCatResults++;
                }
                else
                {
                    incorrectCatResults++;
                }
            }
            foreach (var testDogData in GetImageData("dog", TestDataDir, inputR, inputG, inputB))
            {
                output.PopulateAllOutputs(testDogData);
                if (output.Nodes[0].Output > output.Nodes[1].Output)
                {
                    correctDogResults++;
                }
                else
                {
                    incorrectDogResults++;
                }
            }
            _testOutputHelper.WriteLine($"Cat accuracy: {(double)correctCatResults / correctCatResults + incorrectCatResults}");
            _testOutputHelper.WriteLine($"Dog accuracy: {(double)correctDogResults / correctDogResults + incorrectDogResults}");
            // assert that it can now resolve cat vs dog with other data
        }

        // 1, 0 => cat
        // 0, 1 => dog
        private IEnumerable<Dictionary<Layer, double[]>> GetImageData(string filePrefix, string fileDir, Layer r, Layer g, Layer b)
        {
            foreach (var file in Directory.EnumerateFileSystemEntries(fileDir, $"{filePrefix}*"))
            {
                var image = new Bitmap(file);

                var red = new double[10201];
                var blue = new double[10201];
                var green = new double[10201];
                for (var i = 0; i < 101; i++)
                {
                    for (var j = 0; j < 101; j++)
                    {
                        var pixel = image.GetPixel(i * image.Width / 101, j * image.Height / 101);
                        red[j * 101 + i] = pixel.R;
                        blue[j * 101 + i] = pixel.G;
                        green[j * 101 + i] = pixel.B;
                    }
                }

                yield return new Dictionary<Layer, double[]>()
                {
                    [r] = red,
                    [g] = green,
                    [b] = blue,
                };
            }
        }
    }
}
