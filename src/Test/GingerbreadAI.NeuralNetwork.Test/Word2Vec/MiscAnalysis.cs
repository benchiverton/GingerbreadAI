using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using GingerbreadAI.NLP.Word2Vec;
using GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using GingerbreadAI.NLP.Word2Vec.Extensions;
using Xunit;

namespace GingerbreadAI.NeuralNetwork.Test.Word2Vec
{
    public class MiscAnalysis
    {
        private const string InputFileLoc = "C:\\Temp\\misc.csv";
        private const string ResultsDirectory = nameof(MiscAnalysis);

        [RunnableInDebugOnly]
        public void TrainWordEmbeddings()
        {
            var id = DateTime.Now.Ticks;
            var embeddingsFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/wordEmbeddings-{id}.csv";
            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var word2Vec = new Word2VecTrainer();
            word2Vec.Setup(InputFileLoc, minWordOccurrences: 3);
            word2Vec.TrainModel(useCbow: false, numberOfIterations: 20);
            word2Vec.WriteWordEmbeddings(embeddingsFileLoc);
        }

        [RunnableInDebugOnly]
        public void GenerateDistortionReportForKMeans()
        {
            var embeddingsFile = new DirectoryInfo($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}").EnumerateFiles()
                .Where(f => Regex.IsMatch(f.Name, "^wordEmbeddings-.*$"))
                .OrderBy(f => f.CreationTime)
                .Last();
            var reportFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/report-{DateTime.Now.Ticks}.csv";

            var wordEmbeddings = new List<WordEmbedding>();
            wordEmbeddings.PopulateWordEmbeddingsFromFile(embeddingsFile.FullName);
            wordEmbeddings.NormaliseEmbeddings();

            var articleEmbeddings = new List<ArticleEmbedding>();
            foreach (var line in File.ReadLines(InputFileLoc))
            {
                var splitLine = line.Split(',');
                articleEmbeddings.Add(new ArticleEmbedding(splitLine[0], string.Join(' ', splitLine.Skip(1)), maxContentsLength: 500));
            }
            articleEmbeddings.AssignVectorsFromWeightedWordEmbeddings(wordEmbeddings);

            var kMeans = new KMeans(articleEmbeddings);
            var distortions = new Dictionary<object, object>();
            for (var i = 2; i <= 25; i++)
            {
                kMeans.CalculateLabelClusterMap(numberOfClusters: i);
                distortions.Add(i, kMeans.CalculateDistortion());
            }

            var reportHandler = new ReportWriter(reportFileLoc);
            reportHandler.WriteMisc(distortions);
        }

        [RunnableInDebugOnly]
        public void GenerateReportFromLatestEmbeddings()
        {
            var embeddingsFile = new DirectoryInfo($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}").EnumerateFiles()
                .Where(f => Regex.IsMatch(f.Name, "^wordEmbeddings-.*$"))
                .OrderBy(f => f.CreationTime)
                .Last();
            var reportFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/report-{DateTime.Now.Ticks}.csv";

            var wordEmbeddings = new List<WordEmbedding>();
            wordEmbeddings.PopulateWordEmbeddingsFromFile(embeddingsFile.FullName);
            wordEmbeddings.NormaliseEmbeddings();

            var articleEmbeddings = new List<ArticleEmbedding>();
            foreach (var line in File.ReadLines(InputFileLoc))
            {
                var splitLine = line.Split(',');
                articleEmbeddings.Add(new ArticleEmbedding(splitLine[0], string.Join(' ', splitLine.Skip(1)), maxContentsLength: 500));
            }
            articleEmbeddings.AssignVectorsFromWeightedWordEmbeddings(wordEmbeddings);

            var kMeans = new KMeans(articleEmbeddings);
            kMeans.CalculateLabelClusterMap(
                numberOfClusters: 25
            );

            var reportHandler = new ReportWriter(reportFileLoc);
            reportHandler.WriteLabelsWithClusterIndex(kMeans.LabelClusterMap);
        }
    }
}
