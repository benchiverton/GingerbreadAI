using System;
using System.Linq;

namespace Word2Vec
{
    public static class HuffmanTree
    {
        /**
         * ======== CreateBinaryTree ========
         * Create binary Huffman tree using the word counts.
         * Frequent words will have short unique binary codes.
         * Huffman encoding is used for lossless compression.
         * The vocab_word structure contains a field for the 'code' for the word.
         */
        public static void Create(WordCollection _wordCollection, int maxCodeLength)
        {
            var code = new char[maxCodeLength];
            var point = new long[maxCodeLength];
            var count = new long[_wordCollection.GetNumberOfUniqueWords() * 2 + 1];
            var binary = new long[_wordCollection.GetNumberOfUniqueWords() * 2 + 1];
            var parentNode = new int[_wordCollection.GetNumberOfUniqueWords() * 2 + 1];
            var keys = _wordCollection.GetWords().ToArray();

            for (var a = 0; a < _wordCollection.GetNumberOfUniqueWords(); a++)
                count[a] = _wordCollection.GetOccuranceOfWord(keys[a]);
            for (var a = _wordCollection.GetNumberOfUniqueWords(); a < _wordCollection.GetNumberOfUniqueWords() * 2; a++)
                count[a] = (long)1e15;
            long pos1 = _wordCollection.GetNumberOfUniqueWords() - 1;
            long pos2 = _wordCollection.GetNumberOfUniqueWords();
            for (var a = 0; a < _wordCollection.GetNumberOfUniqueWords() - 1; a++)
            {
                bool decideDirection(long x, long y) => x >= 0 && count[x] < count[y];
                long min1I;
                long min2I;
                (min1I, pos1, pos2) = GetMinI(pos1, pos2, decideDirection);
                (min2I, pos1, pos2) = GetMinI(pos1, pos2, decideDirection);

                count[_wordCollection.GetNumberOfUniqueWords() + a] = count[min1I] + count[min2I];
                parentNode[min1I] = _wordCollection.GetNumberOfUniqueWords() + a;
                parentNode[min2I] = _wordCollection.GetNumberOfUniqueWords() + a;
                binary[min2I] = 1;
            }
            for (long wordIndex = 0; wordIndex < _wordCollection.GetNumberOfUniqueWords(); wordIndex++)
            {
                var b = wordIndex;
                long i = 0;
                while (true)
                {
                    code[i] = (char)binary[b];
                    point[i] = b;
                    i++;
                    b = parentNode[b];
                    if (b == _wordCollection.GetNumberOfUniqueWords() * 2 - 2)
                        break;
                }

                _wordCollection.SetCodeLength(keys, i, wordIndex);
                _wordCollection.SetPoint(keys, wordIndex);
                for (b = 0; b < i; b++)
                {
                    _wordCollection.SetCode(keys, wordIndex, i, b, code);
                    _wordCollection.SetPoint2(keys, wordIndex, i, b, point);
                }
            }
            GC.Collect();
        }

        private static (long MinI, long pos1, long pos2) GetMinI(long pos1, long pos2,
            Func<long, long, bool> decideDirection)
            => decideDirection(pos1, pos2) ? (pos1, pos1 - 1, pos2) : (pos2, pos1, pos2 + 1);
    }
}