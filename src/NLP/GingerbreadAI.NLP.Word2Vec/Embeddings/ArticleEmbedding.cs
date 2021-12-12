namespace GingerbreadAI.NLP.Word2Vec.Embeddings;

public record ArticleEmbedding : IEmbedding
{
    public ArticleEmbedding(string title, string content, int maxContentsLength = 50)
    {
        Label = title;

        Contents = new WordCollection();
        Contents.AddWords(content, maxContentsLength);
    }

    public string Label { get; init; }
    public double[] Vector { get; set; }

    public WordCollection Contents { get; }
}
