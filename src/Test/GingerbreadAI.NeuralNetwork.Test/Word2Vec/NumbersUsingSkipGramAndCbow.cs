using System;
using System.IO;
using GingerbreadAI.NLP.Word2Vec;
using Xunit;

namespace GingerbreadAI.NeuralNetwork.Test.Word2Vec
{
    public class NumbersUsingSkipGramAndCbow
    {
        private const string ResultsDirectory = nameof(NumbersUsingSkipGramAndCbow);

        [RunnableInDebugOnly]
        public void Go()
        {
            var inputFile = $@"{Directory.GetCurrentDirectory()}/TestData/numbers.txt";
            var outputFile = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{DateTime.Now.Ticks}.csv";

            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var fileHandler = new FileHandler(inputFile, outputFile);
            var word2Vec = new Word2VecUsingLibrary(fileHandler, numberOfDimensions: 50, numberOfThreads: 4, numberOfIterations: 4, windowSize: 1, thresholdForOccurrenceOfWords: 0);

            word2Vec.TrainModel();
        }
    }
}
