using System;
using System.IO;
using GingerbreadAI.NLP.Word2Vec;

namespace GingerbreadAI.NeuralNetwork.Test.Word2Vec
{
    public class NumbersUsingCbow
    {
        private const string ResultsDirectory = nameof(NumbersUsingCbow);

        [RunnableInDebugOnly]
        public void Go()
        {
            var inputFile = $@"{Directory.GetCurrentDirectory()}/TestData/numbers.txt";
            var outputFile = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{DateTime.Now.Ticks}.csv";

            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var fileHandler = new FileHandler(inputFile, outputFile);
            var word2Vec = new Word2VecUsingLibrary();
            word2Vec.Setup(fileHandler);

            word2Vec.Train(numberOfThreads: 4, numberOfIterations: 4, windowSize: 1, thresholdForOccurrenceOfWords: 0, useSkipgram: false);

            fileHandler.WriteOutputMatrix(word2Vec.WordCollection, word2Vec.NeuralNetwork);
        }
    }
}
