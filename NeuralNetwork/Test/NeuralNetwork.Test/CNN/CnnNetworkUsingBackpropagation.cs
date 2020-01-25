using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;
using Xunit.Abstractions;

namespace NeuralNetwork.Test.CNN
{
    public class CnnNetworkUsingBackpropagation
    {
        private const string ResultsDirectory = nameof(CnnNetworkUsingBackpropagation);
        private readonly ITestOutputHelper _testOutputHelper;

        public CnnNetworkUsingBackpropagation(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }
    }
}
