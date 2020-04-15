using System;
using System.IO;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.Extensions;
using Xunit;

namespace GingerbreadAI.NeuralNetwork.Test.Word2Vec
{
    public class BlogUsingSkipGramAndCbow
    {
        private const string ResultsDirectory = nameof(BlogUsingSkipGramAndCbow);

        [RunnableInDebugOnly]
        public void Go()
        {
            var inputFileLoc = TrainingDataManager.GetBlogAuthorshipCorpusFiles().First(f => f.Length >= 3e5 && f.Length <= 4e5).FullName;
            var outputFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{DateTime.Now.Ticks}.csv";

            var fileHandler = new FileHandler(inputFileLoc, outputFileLoc);
            var word2Vec = new Word2VecTrainer();
            word2Vec.Setup(fileHandler);

            word2Vec.TrainModel(numberOfIterations: 16);

            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");
            fileHandler.WriteWordClusterLabels(
                word2Vec.WordCollection,
                word2Vec.NeuralNetwork,
                epsilon: 0.25,
                minimumSamples: 3,
                distanceFunctionType: DistanceFunctionType.Cosine);
        }
    }
}
