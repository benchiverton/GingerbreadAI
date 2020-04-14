﻿using System;
using System.IO;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec;
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
            var inputFileLoc = TrainingDataManager.GetBlogAuthorshipCorpusFiles().First(f => f.Length >= 1e5 && f.Length <= 2e5).FullName;
            var outputFileLoc = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{DateTime.Now.Ticks}.csv";

            var fileHandler = new FileHandler(inputFileLoc, outputFileLoc);
            var word2Vec = new Word2VecTrainer();
            word2Vec.Setup(fileHandler);

            word2Vec.TrainModel();

            Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");
            fileHandler.WriteSimilarWords(word2Vec.WordCollection, word2Vec.NeuralNetwork, 3);
        }
    }
}
