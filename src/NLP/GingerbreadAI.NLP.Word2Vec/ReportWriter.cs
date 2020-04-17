using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GingerbreadAI.Model.NeuralNetwork.Models;
using GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;

namespace GingerbreadAI.NLP.Word2Vec
{
    public class ReportWriter
    {
        private readonly string _reportFile;

        public ReportWriter(string reportFile)
        {
            _reportFile = reportFile;
        }

        /// <summary>
        /// Writes each word and the top n words that it is most similar to.
        /// </summary>
        public void WriteSimilarWords(
            IEnumerable<WordEmbedding> wordEmbeddings,
            int topn = 10)
        {
            using (var fs = new FileStream(_reportFile, FileMode.OpenOrCreate, FileAccess.Write))
            {
                fs.Seek(0, SeekOrigin.End);
                using (var writer = new StreamWriter(fs, Encoding.UTF8))
                {
                    foreach (var word in wordEmbeddings.Select(we => we.Word))
                    {
                        var similarWords = WordEmbeddingAnalysisFunctions.GetMostSimilarWords(word, wordEmbeddings, topn);
                        writer.WriteLine($"{word},{string.Join(',', similarWords.Select(sw => $"{sw.word},{sw.similarity:0.00000}"))}");
                    }
                }
            }
        }

        /// <summary>
        /// Writes each word and the top n words that it is most similar to.
        /// </summary>
        public void WriteWordClusterLabels(
            IEnumerable<WordEmbedding> wordEmbeddings,
            double epsilon = 0.5,
            int minimumSamples = 5,
            DistanceFunctionType distanceFunctionType = DistanceFunctionType.Euclidean,
            int concurrentThreads = 4)
        {
            using (var fs = new FileStream(_reportFile, FileMode.OpenOrCreate, FileAccess.Write))
            {
                fs.Seek(0, SeekOrigin.End);
                using (var writer = new StreamWriter(fs, Encoding.UTF8))
                {
                    foreach (var (word, clusterIndex) in WordEmbeddingAnalysisFunctions.GetClusterLabels(
                        wordEmbeddings.ToList(),
                        epsilon,
                        minimumSamples,
                        distanceFunctionType,
                        concurrentThreads))
                    {
                        writer.WriteLine($"{word},{clusterIndex}");
                    }
                }
            }
        }

        /// <summary>
        /// Writes a matrix where x and y axis are the words in the collection, and the point (x, y) is E(x|y).
        /// </summary>
        public void WriteProbabilityMatrix(
            WordCollection wordCollection,
            Layer neuralNetwork)
        {
            using (var fs = new FileStream(_reportFile, FileMode.OpenOrCreate, FileAccess.Write))
            {
                fs.Seek(0, SeekOrigin.End);
                using (var writer = new StreamWriter(fs, Encoding.UTF8))
                {
                    var words = wordCollection.GetWords().ToList();

                    var stringBuilder = new StringBuilder();

                    stringBuilder.Append(",");
                    foreach (var word in words)
                    {
                        stringBuilder.Append($"{word},");
                    }
                    stringBuilder.AppendLine();
                    for (var i = 0; i < words.Count; i++)
                    {
                        var inputs = new double[words.Count];
                        inputs[i] = 1;
                        neuralNetwork.CalculateOutputs(inputs);

                        stringBuilder.Append($"{words[i]},");
                        for (var j = 0; j < words.Count; j++)
                        {
                            stringBuilder.Append($"{neuralNetwork.Nodes[j].Output},");
                        }
                        stringBuilder.AppendLine();
                    }

                    writer.WriteLine(stringBuilder.ToString());
                }
            }
        }
    }
}
