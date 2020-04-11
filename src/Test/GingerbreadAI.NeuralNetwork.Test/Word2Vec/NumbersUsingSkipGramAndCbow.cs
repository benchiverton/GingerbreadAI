using System;
using System.IO;
using GingerbreadAI.NLP.Word2Vec;

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

            System.IO.Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var fileHandler = new FileHandler(inputFile, outputFile);
            var word2Vec = new Word2VecUsingLibrary(fileHandler, numberOfDimensions: 50, numberOfThreads: 6, numberOfIterations: 1, windowSize: 1, thresholdForOccurrenceOfWords: 0);

            word2Vec.TrainModel();
        }
    }
}
