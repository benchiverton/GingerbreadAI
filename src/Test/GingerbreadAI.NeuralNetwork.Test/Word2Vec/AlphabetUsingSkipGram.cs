using System;
using System.IO;
using GingerbreadAI.NLP.Word2Vec;
using GingerbreadAI.NLP.Word2Vec.Extensions;

namespace GingerbreadAI.NeuralNetwork.Test.Word2Vec
{
    public class AlphabetUsingSkipGram
    {
        private const string ResultsDirectory = nameof(AlphabetUsingSkipGram);

        [RunnableInDebugOnly]
        public void Go()
        {
            var inputFileLoc = TrainingDataManager.GetAlphabetFile().FullName;
            var outputFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{DateTime.Now.Ticks}.csv";

            var fileHandler = new FileHandler(inputFileLoc, outputFileLoc);
            var word2Vec = new Word2VecTrainer();
            word2Vec.Setup(fileHandler);

            word2Vec.TrainModel(windowSize: 1, thresholdForOccurrenceOfWords: 0, useCbow: false);

            var wordVectors = word2Vec.WordCollection.GetWordVectors(word2Vec.NeuralNetwork);

            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");
            fileHandler.WriteProbabilityMatrix(word2Vec.WordCollection, word2Vec.NeuralNetwork);
            fileHandler.WriteWordVectors(wordVectors);
        }
    }
}
