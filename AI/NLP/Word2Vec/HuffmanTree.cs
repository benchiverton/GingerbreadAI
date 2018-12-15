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
                count[a] = (long) 1e15;
            long pos1 = _wordCollection.GetNumberOfUniqueWords() - 1;
            long pos2 = _wordCollection.GetNumberOfUniqueWords();
            for (var a = 0; a < _wordCollection.GetNumberOfUniqueWords() - 1; a++)
            {
                long min1I;
                if (pos1 >= 0)
                {
                    if (count[pos1] < count[pos2])
                    {
                        min1I = pos1;
                        pos1--;
                    }
                    else
                    {
                        min1I = pos2;
                        pos2++;
                    }
                }
                else
                {
                    min1I = pos2;
                    pos2++;
                }
                long min2I;
                if (pos1 >= 0)
                {
                    if (count[pos1] < count[pos2])
                    {
                        min2I = pos1;
                        pos1--;
                    }
                    else
                    {
                        min2I = pos2;
                        pos2++;
                    }
                }
                else
                {
                    min2I = pos2;
                    pos2++;
                }
                count[_wordCollection.GetNumberOfUniqueWords() + a] = count[min1I] + count[min2I];
                parentNode[min1I] = _wordCollection.GetNumberOfUniqueWords() + a;
                parentNode[min2I] = _wordCollection.GetNumberOfUniqueWords() + a;
                binary[min2I] = 1;
            }
            for (long a = 0; a < _wordCollection.GetNumberOfUniqueWords(); a++)
            {
                var b = a;
                long i = 0;
                while (true)
                {
                    code[i] = (char) binary[b];
                    point[i] = b;
                    i++;
                    b = parentNode[b];
                    if (b == _wordCollection.GetNumberOfUniqueWords() * 2 - 2)
                        break;
                }
                _wordCollection.SetPoint(keys, a);
                for (b = 0; b < i; b++)
                {
                    _wordCollection.SetCode(keys, a, i, b, code);
                    _wordCollection.SetPoint2(keys, a, i, b, point);
                }
            }
            GC.Collect();
        }
    }
}