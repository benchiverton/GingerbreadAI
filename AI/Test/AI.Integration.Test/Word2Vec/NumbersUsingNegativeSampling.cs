using System;
using System.IO;
using NLP.Word2Vec;

namespace AI.Integration.Test.Word2Vec
{
    public class NumbersUsingNegativeSampling
    {
        private const string ResultsDirectory = nameof(NumbersUsingNegativeSampling);

        [RunnableInDebugOnly]
        public void Go()
        {
            var inputFile = $@"{Directory.GetCurrentDirectory()}/Data/numbers.txt";
            var outputFile = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{DateTime.Now.Ticks}.csv";

            System.IO.Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var fileHandler = new FileHandler(inputFile, outputFile);
            var word2Vec = new Word2VecUsingLibrary(fileHandler, numberOfDimensions: 50, numberOfThreads: 6, numberOfIterations: 1, windowSize: 1, thresholdForOccurrenceOfWords: 0, negativeSamples: 3);

            word2Vec.TrainModel();
        }
    }
}
