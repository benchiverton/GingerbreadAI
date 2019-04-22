namespace Network.Console
{
    using System;
    using Word2Vec.Ben;

    public class Program
    {
        public static void Main()
        {
            var word2Vec = new Word2Vec("input.txt", "wordDictionaryFile.dic", 4, 1);

            word2Vec.TrainModel();
        }
    }
}