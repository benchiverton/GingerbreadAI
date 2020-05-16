namespace GingerbreadAI.NLP.Word2Vec.Embeddings
{
    public interface IEmbedding
    {
        string Label { get; }
        double[] Vector { get; set; }
    }
}
