namespace Emotion.Detector.Repositories.Cache
{
    using System.Collections.Generic;
    using Data;
    using Extensions;
    using log4net;

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
            var isInCache = false;
            emotionData = null;

            if (_unfoundWords.Contains(word))
            {
                isInCache = true;
            }
            else if (_foundWords.ContainsKey(word))
            {
                emotionData = _foundWords[word].CloneJson();
                isInCache = true;
            }

            return isInCache;
        }

        public void AddUnfoundWordToCache(string word)
        {
            lock (_unfoundWordsLock)
            {
                _unfoundWords.Add(word);
            }
        }

        public void AddFoundWordToCache(string word, EmotionData emotionData)
        {
            lock (_foundWordsLock)
            {
                _foundWords.Add(word, emotionData);
            }
        }
    }
}
