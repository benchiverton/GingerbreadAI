using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using DeepLearning.Backpropagation;
using Model.ConvolutionalNeuralNetwork.Extensions;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork;
using Model.NeuralNetwork.Models;
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
            var inputR = new Layer2D((100, 100), new Layer[0]);
            var inputG = new Layer2D((100, 100), new Layer[0]);
            var inputB = new Layer2D((100, 100), new Layer[0]);
            var filters = (new[] { inputR, inputG, inputB }).Add2DConvolutionalLayer(32, 3);
            var pooling = filters.AddPooling(3);
            var stepDownLayer = new Layer(32, pooling.ToArray());
            var output = new Layer(2, new[] { stepDownLayer });

            var trainingDataCat = GetImageData("cat", TrainingDataDir, inputR, inputG, inputB).GetEnumerator();
            var trainingDataDog = GetImageData("dog", TrainingDataDir, inputR, inputG, inputB).GetEnumerator();
            do
            {
                output.Backpropagate(trainingDataCat.Current, new[] { 1d, 0d }, 0.1);
                output.Backpropagate(trainingDataDog.Current, new[] { 0d, 1d }, 0.1);
            } while (trainingDataCat.MoveNext() && trainingDataDog.MoveNext());

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
            var image = Image.FromFile($"{filePrefix}{fileDir}");

            //image.get

            yield return new Dictionary<Layer, double[]>()
            {
                [r] = new double[0],
                [g] = new double[0],
                [b] = new double[0],
            };
        }
    }
}
