using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GingerbreadAI.Model.NeuralNetwork.Models;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using GingerbreadAI.NLP.Word2Vec.Extensions;

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
        /// Writes the label of each embedding and the labels of the top n embeddings that it is most similar to.
        /// </summary>
        public void WriteSimilarEmbeddings(IEnumerable<IEmbedding> embeddings, int topn = 10)
        {
            using (var fs = new FileStream(_reportFile, FileMode.OpenOrCreate, FileAccess.Write))
            {
                fs.Seek(0, SeekOrigin.End);
                using (var writer = new StreamWriter(fs, Encoding.UTF8))
                {
                    embeddings = embeddings.ToArray();
                    foreach (var word in embeddings)
                    {
                        var similarEmbeddings = embeddings.GetMostSimilarEmbeddings(word, topn);
                        writer.WriteLine($"{word},{string.Join(',', similarEmbeddings.Select(sw => $"{sw.embedding.Label},{sw.similarity:0.00000}"))}");
                    }
                }
            }
        }

        /// <summary>
        /// Writes each embedding and the index of the cluster that it is in.
        /// </summary>
        public void WriteLabelsWithClusterIndex(Dictionary<string, int> labelClusterIndexMap, IEnumerable<string> labels)
        {
            using (var fs = new FileStream(_reportFile, FileMode.OpenOrCreate, FileAccess.Write))
            {
                fs.Seek(0, SeekOrigin.End);
                using (var writer = new StreamWriter(fs, Encoding.UTF8))
                {
                    foreach (var label in labels)
                    {
                        writer.WriteLine($"\"{label}\",{labelClusterIndexMap[label]}");
                    }
                }
            }
        }

        /// <summary>
        /// Writes each label, its 2D embedding and its cluster index in a format compatible with excel graphs.
        /// </summary>
        public void Write2DWordEmbeddingsAndClusterIndexesForExcel(List<WordEmbedding> wordEmbeddings, Dictionary<string, int> wordClusterLabels)
        {
            if (wordEmbeddings.First().Vector.Length != 2)
            {
                throw new Exception($"The embeddings ({wordEmbeddings.First().Vector.Length}) needs to be 2 dimensional to use this method.");
            }

            var clusterColumns = wordClusterLabels
                .Select(l => l.Value)
                .Distinct()
                .ToDictionary(clusterId => clusterId, clusterId => 0d);

            using (var fs = new FileStream(_reportFile, FileMode.OpenOrCreate, FileAccess.Write))
            {
                fs.Seek(0, SeekOrigin.End);
                using (var writer = new StreamWriter(fs, Encoding.UTF8))
                {
                    writer.WriteLine($"Word,X,{string.Join(',', clusterColumns.Keys.Select(v => $"Cluster {v}"))}");
                    foreach (var embedding in wordEmbeddings)
                    {
                        var clusterLabel = wordClusterLabels[embedding.Label];
                        clusterColumns[clusterLabel] = embedding.Vector[1];
                        writer.WriteLine($"{embedding.Label},{embedding.Vector[0]},{string.Join(',', clusterColumns.Values.Select(v => v == 0 ? "" : v.ToString()))}");
                        clusterColumns[clusterLabel] = 0;
                    }
                }
            }
        }

        /// <summary>
        /// Writes a matrix where x and y axis are the words in the collection, and the point (x, y) is E(x|y).
        /// </summary>
        public void WriteProbabilityMatrix(WordCollection wordCollection, Layer neuralNetwork)
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
