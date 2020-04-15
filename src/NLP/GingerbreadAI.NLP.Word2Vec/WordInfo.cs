namespace GingerbreadAI.NLP.Word2Vec
{
    public class WordInfo
    {
        public WordInfo(char[] code, long[] point, long position)
            => (Code, Point, Position, Count) = (code, point, position, 1);

        public char[] Code { get; }
        public long[] Point { get; }
        public long Position { get; set; }
        public int Count { get; private set; }
        public int CodeLength { get; set; }

        public void IncrementCount() => Count++;
    }
}