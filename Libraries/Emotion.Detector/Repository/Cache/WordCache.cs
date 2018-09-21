namespace Emotion.Detector.Repository.Cache
{
    using System.Collections.Generic;
    using Data;

    public class WordCache
    {
        private readonly List<string> _unfoundWords;
        private readonly Dictionary<string, Emotion> _foundWords;

        public WordCache()
        {
            _unfoundWords = new List<string>();
            _foundWords = new Dictionary<string, Emotion>();
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
                emotion = _foundWords[word];
            }

            return isInCache;
        }

        public void AddUnfoundWordToCache(string word)
        {
            _unfoundWords.Add(word);
        }

        public void AddFoundWordToCache(string word, Emotion emotion)
        {
            _foundWords.Add(word, emotion);
        }
    }
}
