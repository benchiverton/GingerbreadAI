namespace Word2Vec
{
    public class WordInfo
    {
        public WordInfo(char[] code, int[] point, long position)
            => (Code, Point, Position, Count) = (code, point, position, 1);

        public char[] Code { get; }
        public string ActualWord { get; }
        public int[] Point { get; }
        public long Position { get; set; }
        public long Count { get; private set; }
        public int CodeLength { get; set; }

        public void IncrementCount() => Count++;
    }
}