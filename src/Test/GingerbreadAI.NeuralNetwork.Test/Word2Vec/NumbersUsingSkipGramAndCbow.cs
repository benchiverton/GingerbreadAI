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
            var word2Vec = new Word2VecUsingLibrary();
            word2Vec.Setup(fileHandler, dimensions: 5);

            word2Vec.Train(numberOfThreads: 4, numberOfIterations: 16, windowSize: 1, thresholdForOccurrenceOfWords: 0, negativeSamples: 2);

            fileHandler.WriteOutputMatrix(word2Vec.WordCollection, word2Vec.NeuralNetwork);
        }
    }
}
