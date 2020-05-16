namespace GingerbreadAI.NLP.Word2Vec.Embeddings
{
    public class WordEmbedding : IEmbedding
    {
        public WordEmbedding(string word, double[] vector)
        {
            Label = word;
            Vector = vector;
        }

        public string Label { get; }
        public double[] Vector { get; set; }
    }
}
