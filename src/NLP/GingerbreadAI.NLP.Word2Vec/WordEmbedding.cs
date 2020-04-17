namespace GingerbreadAI.NLP.Word2Vec
{
    public struct WordEmbedding
    {
        public string Word { get; }
        public double[] Vector { get; }

        public WordEmbedding(string word, double[] vector)
        {
            Word = word;
            Vector = vector;
        }
    }
}
