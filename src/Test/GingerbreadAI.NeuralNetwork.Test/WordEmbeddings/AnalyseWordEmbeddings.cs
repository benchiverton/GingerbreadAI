using System;
using System.Collections.Generic;
using System.IO;
using GingerbreadAI.NLP.Word2Vec;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.Extensions;

namespace GingerbreadAI.NeuralNetwork.Test.WordEmbeddings
{
    public class AnalyseWordEmbeddings
    {
        private const string ResultsDirectory = nameof(AnalyseWordEmbeddings);

        [RunnableInDebugOnly]
        public void Go()
        {
            var embeddingsFileLoc = @"C:\Projects\AI\GingerbreadAI\src\Test\GingerbreadAI.NeuralNetwork.Test\bin\Debug\netcoreapp3.1\BlogUsingSkipGramAndCbow\wordEmbeddings-637227200041708622.csv";
            var reportFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/report-{DateTime.Now.Ticks}.csv";
            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var wordEmbeddings = new List<WordEmbedding>();
            wordEmbeddings.PopulateWordEmbeddingsFromFile(embeddingsFileLoc);

            var reportHandler = new ReportWriter(reportFileLoc);
            reportHandler.WriteWordClusterLabels(
                wordEmbeddings,
                epsilon: 0.25,
                minimumSamples: 3,
                distanceFunctionType: DistanceFunctionType.Cosine,
                concurrentThreads: 4);
        }
    }
}
