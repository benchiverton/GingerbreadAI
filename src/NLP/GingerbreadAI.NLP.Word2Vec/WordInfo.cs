namespace GingerbreadAI.NLP.Word2Vec;

public record WordInfo
{
    public WordInfo(char[] code, long[] point, long position)
        => (Code, Point, Position, Count) = (code, point, position, 1);

    public char[] Code { get; init; }
    public long[] Point { get; init; }
    public long Position { get; set; }
    public int Count { get; private set; }
    public int CodeLength { get; set; }

    public void IncrementCount() => Count++;
}
