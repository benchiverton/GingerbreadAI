using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GingerbreadAI.NLP.Word2Vec;
using GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
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
            word2Vec.TrainModel();
            word2Vec.WriteWordEmbeddings(embeddingsFileLoc);


            using var fileStream = new FileStream(embeddingsFileLoc, FileMode.OpenOrCreate, FileAccess.Read);
            using var reader = new StreamReader(fileStream, Encoding.UTF8);
            var wordEmbeddings = new List<WordEmbedding>();
            wordEmbeddings.PopulateWordEmbeddingsFromStream(reader);

            var tsne = new TSNE(2, distanceFunctionType: DistanceFunctionType.Cosine);
            tsne.ReduceDimensions(wordEmbeddings);

            var labelClusterIndexMap = DBSCAN.GetLabelClusterMap(
                wordEmbeddings,
                epsilon: 0.1,
                minimumSamples: 3,
                distanceFunctionType: DistanceFunctionType.Cosine,
                concurrentThreads: 4);

            var reportHandler = new ReportWriter(reportFileLoc);
            reportHandler.Write2DWordEmbeddingsAndClusterIndexesForExcel(wordEmbeddings, labelClusterIndexMap);
        }
    }
}
