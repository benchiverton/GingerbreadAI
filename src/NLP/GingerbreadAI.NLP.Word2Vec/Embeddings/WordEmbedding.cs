namespace GingerbreadAI.NLP.Word2Vec.Embeddings;

public record WordEmbedding : IEmbedding
{
    public WordEmbedding(string word, double[] vector)
    {
        Label = word;
        Vector = vector;
    }

    public string Label { get; init; }
    public double[] Vector { get; set; }
}
