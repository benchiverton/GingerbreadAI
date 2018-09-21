namespace Emotion.Detector
{
    using System.Collections.Generic;
    using System.Linq;
    using Data;
    using Repository;
    using Repository.Cache;

    public class EmotionDetector
    {
        private readonly WordRepository _repository;

        public EmotionDetector(WordRepository repository)
        {
            _repository = repository;
        }

        public Emotion Detect(string text)
        {
            var words = text.Split(' ').ToList();
            var emotions = _repository.GetEmotions(words);

            AmendNegations(emotions);

            return AggregateEmotions(emotions.Select(e => e.emotion).Where(e => e != null).ToList());
        }

        // Don't worry, this will DEFINITELY detect sarcasm.
        public void AmendNegations(List<(string  word, Emotion emotion)> emotions)
        {
            for (var i = 1; i < emotions.Count; i++)
            {
                if (emotions[i].emotion == null) continue;

                if (emotions[i - 1].word.DetectNegation())
                {
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
