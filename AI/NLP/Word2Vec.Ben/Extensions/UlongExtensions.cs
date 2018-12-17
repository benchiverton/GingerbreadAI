namespace Word2Vec.Ben.Extensions
{
    public static class UlongExtensions
    {
        public static ulong LinearCongruentialGenerator(this ulong nextRandom) => nextRandom * 25214903917 + 11;
    }
}
