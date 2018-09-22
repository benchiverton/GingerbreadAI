namespace Emotion.Detector.Repository.Cache
{
    using System.Collections.Generic;
    using Data;
    using Emotion.Detector.Extensions;
    using log4net;

    public class WordCache
    {
        private readonly ILog _log;

        private readonly List<string> _unfoundWords;
        private readonly Dictionary<string, Emotion> _foundWords;

        private readonly object _unfoundWordsLock;
        private readonly object _foundWordsLock;

        public WordCache(ILog log)
        {
            _log = log;

            _unfoundWords = new List<string>();
            _foundWords = new Dictionary<string, Emotion>();

            _unfoundWordsLock = new object();
            _foundWordsLock = new object();
        }

        public bool TryGetWordFromCache(string word, out Emotion emotion)
        {
            var isInCache = false;
            emotion = null;

            if (_unfoundWords.Contains(word))
            {
                isInCache = true;
            }
            else if (_foundWords.ContainsKey(word))
            {
                emotion = _foundWords[word].CloneJson();
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
            _log.Debug($"Unfound word '{word}' saved to cache.");
        }

        public void AddFoundWordToCache(string word, Emotion emotion)
        {
            lock (_foundWordsLock)
            {
                _foundWords.Add(word, emotion);
            }
            _log.Debug($"Found word '{word}' saved to cache.");
        }
    }
}
