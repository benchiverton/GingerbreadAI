using System;
using System.Linq;
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

namespace GingerbreadAI.NeuralNetwork.Test.CNN
{
    public class Cnn2dLettersUsingBackpropagation
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public Cnn2dLettersUsingBackpropagation(ITestOutputHelper testOutputHelper) => _testOutputHelper = testOutputHelper;

        [RunnableInDebugOnly]
        public void TrainAgainstHandWrittenNumbers()
        {
            var input = new Layer2D((28, 28), Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var filters = new[] { input }.Add2DConvolutionalLayer(32, (3, 3), ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            filters.AddPooling((2, 2));
            var stepDownLayer = new Layer(100, filters.ToArray(), ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(10, new[] { stepDownLayer }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.GlorotUniform);
            output.AddMomentumRecursively();
            output.Initialise(new Random());

            foreach (var (image, label) in TrainingDataManager.GetMNISTHandwrittenNumbers("train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz"))
            {
                var targetOutputs = new double[10];
                targetOutputs[label] = 1d;
                output.Backpropagate(image, targetOutputs, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
            }

            var correctResults = new double[10];
            var incorrectResults = new double[10];
            foreach (var (image, label) in TrainingDataManager.GetMNISTHandwrittenNumbers("t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"))
            {
                output.CalculateOutputs(image);
                if (output.Nodes[label].Output > 0.5)
                {
                    correctResults[label]++;
                }
                else
                {
                    incorrectResults[label]++;
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
    }
}
