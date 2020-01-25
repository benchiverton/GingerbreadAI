using System;
using System.ComponentModel.Design.Serialization;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;

namespace Model.ConvolutionalNeuralNetwork.Test.Models
{
    public class FilterShould
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public FilterShould(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void ThrowAnExceptionWhenInputIsInvalid()
        {
        }
    }
}