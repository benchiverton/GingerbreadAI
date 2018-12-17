namespace Word2Vec
{
    internal class WordInfo
    {
        public WordInfo(char[] code, int[] point, long position)
            => (Code, Point, Position, Count) = (code, point, position, 1);

        public char[] Code { get; }
        public int[] Point { get; }
        public long Position { get; set; }
        public long Count { get; private set; }

        public void IncrementCount() => Count++;
    }
}