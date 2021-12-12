using System;
using System.IO;
using GingerbreadAI.NLP.Word2Vec;

namespace GingerbreadAI.NeuralNetwork.Test.Word2Vec;

public class AlphabetUsingSkipGramAndCbow
{
    private const string ResultsDirectory = nameof(AlphabetUsingSkipGramAndCbow);

    [RunnableInDebugOnly]
    public void Go()
    {
        var id = DateTime.Now.Ticks;
        var inputFileLoc = TrainingDataManager.GetAlphabetFile().FullName;
        var embeddingsFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/wordEmbeddings-{id}.csv";
        var reportFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/report-{id}.csv";
        Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

        var word2Vec = new Word2VecTrainer();
        word2Vec.Setup(inputFileLoc);
        word2Vec.TrainModel(windowSize: 1, thresholdForOccurrenceOfWords: 0, useSkipgram: false);
        word2Vec.WriteWordEmbeddings(embeddingsFileLoc);

        var reportHandler = new ReportWriter(reportFileLoc);
        reportHandler.WriteProbabilityMatrix(word2Vec.WordCollection, word2Vec.NeuralNetwork);
    }
}
