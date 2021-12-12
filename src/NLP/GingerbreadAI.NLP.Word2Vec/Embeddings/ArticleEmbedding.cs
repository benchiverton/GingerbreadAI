namespace GingerbreadAI.NLP.Word2Vec.Embeddings;

public class ArticleEmbedding : IEmbedding
{
    public ArticleEmbedding(string title, string content, int maxContentsLength = 50)
    {
        Label = title;

        Contents = new WordCollection();
        Contents.AddWords(content, maxContentsLength);
    }

    public string Label { get; }
    public double[] Vector { get; set; }

    public WordCollection Contents { get; }
}
