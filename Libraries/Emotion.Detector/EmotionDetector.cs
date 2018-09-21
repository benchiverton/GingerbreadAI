namespace Emotion.Detector
{
    using System.Collections.Generic;
    using System.Linq;
    using Data;
    using Emotion.Detector.Extensions;
    using log4net;
    using Repository;
    using Repository.Cache;

    public class EmotionDetector
    {
        private readonly ILog _log;
        private readonly WordRepository _repository;

        public EmotionDetector(ILog log, WordRepository repository)
        {
            _log = log;
            _repository = repository;
        }

        public Emotion Detect(string text)
        {
            var words = text.Split(' ').ToList();
            var emotions = _repository.GetEmotions(words);

            AmendNegations(emotions);

            var foundEmotions = emotions.Where(e => e.emotion != null);
            _log.Debug($"{foundEmotions.Count()} words found with an emotion. These words are:\n {foundEmotions.Select(e => $"e.word\n")}");
            return AggregateEmotions(foundEmotions.Select(e => e.emotion).ToList());
        }

        // Don't worry, this will DEFINITELY detect sarcasm.
        public void AmendNegations(List<(string word, Emotion emotion)> emotions)
        {
            for (var i = 1; i < emotions.Count; i++)
            {
                if (emotions[i].emotion == null) continue;

                if (emotions[i - 1].word.DetectNegation())
                {
                    _log.Debug($"Negation detected ('{emotions[i - 1].word}'). Emotion for {emotions[i].word} is being inverted.");
                    emotions[i].emotion.Invert();
                }
            }
        }

        public Emotion AggregateEmotions(List<Emotion> emotions)
        {
            return new Emotion
            {
                Anger = emotions.Select(e => e.Anger).Average(),
                Disgust = emotions.Select(e => e.Disgust).Average(),
                Fear = emotions.Select(e => e.Fear).Average(),
                Happiness = emotions.Select(e => e.Happiness).Average(),
                Sadness = emotions.Select(e => e.Sadness).Average(),
                Surprise = emotions.Select(e => e.Surprise).Average()
            };
        }
    }
}
