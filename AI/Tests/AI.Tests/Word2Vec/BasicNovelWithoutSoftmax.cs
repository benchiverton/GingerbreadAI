namespace AI.Tests.Word2Vec
{
    using System;
    using System.IO;
    using global::Word2Vec.Ben;
    using Xunit;

    public class BasicNovelWithoutSoftmax
    {
        private const string ResultsDirectory = nameof(BasicNovelWithoutSoftmax);

        [RunnableInDebugOnly]
        public void Go()
        {
            var inputFile = $@"{Directory.GetCurrentDirectory()}/Data/HP_1_C1.txt";
            var outputFile = $@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}/networkResults-{DateTime.Now.Ticks}.csv";

            System.IO.Directory.CreateDirectory($@"{Directory.GetCurrentDirectory()}/{ResultsDirectory}");

            var word2Vec = new Word2Vec(inputFile, outputFile, numberOfIterations: 20, 
                useHs: false);

            word2Vec.TrainModel();
        }
    }
}
