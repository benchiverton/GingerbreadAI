using System;
using System.IO;
using GingerbreadAI.NLP.Word2Vec;
using GingerbreadAI.NLP.Word2Vec.Extensions;
using Xunit;

namespace GingerbreadAI.NeuralNetwork.Test.Word2Vec
{
    public class AlphabetUsingSkipGramAndCbow
    {
        private const string ResultsDirectory = nameof(AlphabetUsingSkipGramAndCbow);

        [RunnableInDebugOnly]
        public void Go()
        {
            var inputFileLoc = TrainingDataManager.GetAlphabetFile().FullName;
            var embeddingsFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/wordEmbeddings-{DateTime.Now.Ticks}.csv";
            var reportFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/report-{DateTime.Now.Ticks}.csv";
            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");
            
            var word2Vec = new Word2VecTrainer();
            word2Vec.Setup(inputFileLoc);
            word2Vec.TrainModel(windowSize: 1, thresholdForOccurrenceOfWords: 0, useSkipgram: false);
            word2Vec.WriteWordEmbeddings(embeddingsFileLoc);

            var reportHandler = new ReportWriter(reportFileLoc);
            reportHandler.WriteProbabilityMatrix(word2Vec.WordCollection, word2Vec.NeuralNetwork);
        }
    }
}
