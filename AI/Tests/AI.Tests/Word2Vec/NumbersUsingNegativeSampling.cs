namespace AI.Tests.Word2Vec
{
    using global::Word2Vec;
    using NeuralNetwork.Models;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;

    public class NumbersUsingNegativeSampling
    {
        private const string ResultsDirectory = nameof(NumbersUsingNegativeSampling);

        [RunnableInDebugOnly]
        public void Go()
        {
            var inputFile = $@"{Directory.GetCurrentDirectory()}/Data/numbers.txt";
            var outputFile = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{DateTime.Now.Ticks}.csv";

            System.IO.Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var windowSize = 1;
            var word2Vec = new Word2VecUsingLibrary(inputFile, outputFile, numberOfDimensions: 50, numberOfThreads: 4, numberOfIterations: 1, windowSize: windowSize, thresholdForOccurrenceOfWords: 0, negative: 3);

            word2Vec.TrainModel();
        }
    }
}
