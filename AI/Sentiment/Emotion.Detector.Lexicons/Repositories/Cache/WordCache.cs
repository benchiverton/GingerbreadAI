using System;
using System.Collections.Generic;
using Emotion.Detector.Lexicons.Data;
using Emotion.Detector.Lexicons.Extensions;
using log4net;

namespace Emotion.Detector.Lexicons.Repositories.Cache
{
    public class WordCache
    {
        private readonly ILog _log;

        private readonly List<string> _unfoundWords;
        private readonly Dictionary<string, EmotionData> _foundWords;

        private readonly object _unfoundWordsLock;
        private readonly object _foundWordsLock;

        public WordCache(ILog log)
        {
            _log = log;

            _unfoundWords = new List<string>();
            _foundWords = new Dictionary<string, EmotionData>();

            _unfoundWordsLock = new object();
            _foundWordsLock = new object();
        }

        public bool TryGetWordFromCache(string word, out EmotionData emotionData)
        {
            emotionData = null;

            lock (_foundWords)
            {
                if (_foundWords.ContainsKey(word))
                {
                    emotionData = _foundWords[word].CloneJson();
                    return true;
                }
            }

            lock (_unfoundWordsLock)
            {
                if (_unfoundWords.Contains(word))
                {
                    return true;
                }
            }

            return false;
        }

        public void AddUnfoundWordToCache(string word)
        {
            lock (_unfoundWordsLock)
            {
                try
                {
                    _unfoundWords.Add(word);
                }
                catch (Exception)
                {
                    // do nothing
                }
            }
        }

        public void AddFoundWordToCache(string word, EmotionData emotionData)
        {
            lock (_foundWordsLock)
            {
                _foundWords.TryAdd(word, emotionData);
            }
        }
    }
}
