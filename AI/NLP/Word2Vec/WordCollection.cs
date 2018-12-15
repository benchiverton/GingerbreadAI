using System;
using System.Collections.Generic;
using System.Linq;

namespace Word2Vec
{
    public class WordCollection
    {
        private readonly Dictionary<string, long> _countPerWord;
        private readonly Dictionary<string, char[]> _code;
        private readonly Dictionary<string, int[]> _point;
        private readonly Dictionary<string, long?> _wordPosition;

        public long? this[string index] => _wordPosition.ContainsKey(index) ? _wordPosition[index] : null;

        public WordCollection()
        {
            _countPerWord = new Dictionary<string, long>();
            _code = new Dictionary<string, char[]>();
            _point = new Dictionary<string, int[]>();
            _wordPosition = new Dictionary<string, long?>();
            
        }

        public void AddWords(string line, int maxCodeLength)
        {
                var words = line.Split(new[] { "\r", "\n", "\t", " ", ",", "\"" },
                    StringSplitOptions.RemoveEmptyEntries).Select(y => y.ToLower());

                foreach (var word in words)
                {
                    if (string.IsNullOrWhiteSpace(word))
                        continue;

                    var cleanWord = Clean(word);
                    if (ContainsWord(cleanWord))
                    {
                        _countPerWord[cleanWord]++;
                    }
                    else
                    {
                        AddWordToDictionary(cleanWord, 1, maxCodeLength);
                    }
                }
            }

        public static string Clean(string input)
        {
            return input.Replace("\r", " ").Replace("\n", " ").Replace("\t", " ").Replace(",", " ").Replace("\"", " ").ToLower();
        }

        private void AddWordToDictionary(string word, long wordCount, int MaxCodeLength)
        {
            _countPerWord.Add(word, wordCount);
            _code.Add(word, new char[MaxCodeLength]);
            _point.Add(word, new int[MaxCodeLength]);
        }

        public void InitWordPositions()
        {
            foreach (var x in GetWords())
            {
                _wordPosition.Add(x, _wordPosition.Count);
            }
        }

        public int GetNumberOfUniqueWords()
        {
            return _countPerWord.Count;
        }

        public IEnumerable<string> GetWords()
        {
            return _countPerWord.Keys;
        }

        public long GetTotalNumberOfWords()
        {
            return _countPerWord.Sum(x => x.Value);
        }

        public double GetTrainWordsPow(double power)
        {
            return _countPerWord.Sum(x => Math.Pow(x.Value, power));
        }

        public long GetOccuranceOfWord(string word)
        {
            return _countPerWord[word];
        }

        public bool ContainsWord(string word)
        {
            return _countPerWord.ContainsKey(word);
        }

        public void RemoveWordsWithCountLessThanMinCount(int minCount)
        {
            var remove = (from x in _countPerWord where x.Value < minCount select x.Key).ToArray();
            foreach (var x in remove)
            {
                _countPerWord.Remove(x);
                _code.Remove(x);
                _point.Remove(x);
            }
            GC.Collect();
        }


        public void SetPoint2(string[] keys, long a, long i, long b, long[] point)
        {
            _point[keys[a]][i - b] = (int)(point[b] - GetNumberOfUniqueWords());
        }

        public void SetPoint(string[] keys, long a)
        {
            _point[keys[a]][0] = GetNumberOfUniqueWords() - 2;
        }

        public void SetCode(string[] keys, long a, long i, long b, char[] code)
        {
            _code[keys[a]][i - b - 1] = code[b];
        }
    }
}