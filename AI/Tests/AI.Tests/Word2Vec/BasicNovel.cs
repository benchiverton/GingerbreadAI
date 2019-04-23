namespace AI.Tests.Word2Vec
{
    using global::Word2Vec.Ben;
    using Xunit;

    public class BasicNovel
    {
        [RunnableInDebugOnly]
        public void PerformWord2Vec()
        {
            var word2Vec = new Word2Vec(@"C:\Users\benc\Downloads\Harry Potter and the Sorcerer.txt", @"C:\Users\benc\Downloads\outputFile.csv", useHs: true, numberOfIterations: 20);

            word2Vec.TrainModel();
        }
    }
}
