using System;
using System.Collections.Generic;
using System.Text;

namespace Word2Vec.Extensions
{
    public static class UlongExtensions
    {
        public static ulong LinearCongruentialGenerator(this ulong nextRandom) => nextRandom * 25214903917 + 11;
    }
}
