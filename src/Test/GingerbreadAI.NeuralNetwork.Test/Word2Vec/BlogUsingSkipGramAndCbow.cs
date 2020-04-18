using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.Extensions;

namespace GingerbreadAI.NeuralNetwork.Test.Word2Vec
{
    public class BlogUsingSkipGramAndCbow
    {
        private const string ResultsDirectory = nameof(BlogUsingSkipGramAndCbow);

        [RunnableInDebugOnly]
        public void Go()
        {
            var id = DateTime.Now.Ticks;
            var inputFileLoc = TrainingDataManager.GetBlogAuthorshipCorpusFiles().First(f => f.Length >= 1e5 && f.Length <= 2e5).FullName;
            var embeddingsFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/wordEmbeddings-{id}.csv";
            var reportFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/report-{id}.csv";
            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var word2Vec = new Word2VecTrainer();
            word2Vec.Setup(inputFileLoc, minWordOccurrences: 3);
            word2Vec.TrainModel(numberOfIterations: 16);
            word2Vec.WriteWordEmbeddings(embeddingsFileLoc);

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
