namespace Emotion.Detector
{
    using System.Collections.Generic;
    using System.Linq;
    using Data;
    using Emotion.Detector.Extensions;
    using log4net;
    using Repositories;
    using Repositories.Cache;

    public class EmotionDetector
    {
        private readonly ILog _log;
        private readonly WordRepository _repository;
        private readonly NegationManager _negationManager;

        public EmotionDetector(ILog log, WordRepository repository, NegationManager negationManager)
        {
            _log = log;
            _repository = repository;
            _negationManager = negationManager;
        }

        public Emotion Detect(string text)
        {
            var words = text.GetWordsFromText();
            var emotions = _repository.GetEmotions(words);

            AmendNegations(emotions);

            var foundEmotions = emotions.Where(e => e.emotion != null);
            _log.Debug($"{foundEmotions.Count()} words found with an emotion in the text '{text}'. These words are:\n {foundEmotions.Select(x => x.word).Aggregate((w1, w2) => $"{w1}\r\n{w2}")}");
            return foundEmotions.Select(e => e.emotion).GetOverallEmotion();
        }

        // Don't worry, this will DEFINITELY detect sarcasm.
        private void AmendNegations(List<(string word, Emotion emotion)> emotions)
        {
            for (var i = 1; i < emotions.Count; i++)
            {
                if (emotions[i].emotion == null) continue;

                if (_negationManager.IsNegation(emotions[i - 1].word))
                {
                    _log.Debug($"Negation detected ('{emotions[i - 1].word}'). Emotion for {emotions[i].word} is being inverted.");
                    emotions[i].emotion.Invert();
                }
            }
        }
    }
}
