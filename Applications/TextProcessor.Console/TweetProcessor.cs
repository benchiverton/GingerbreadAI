using Emotion.Detector;
using log4net;
using System.Linq;
using TwitterProcessor.Console.Data;

namespace TwitterProcessor.Console
{
    using Emotion.Detector.Data;

    // process emotion
    // persist tweet
    public class TweetProcessor
    {
        private readonly ILog _log;
        private readonly EmotionDetector _emotionDetector;

        public TweetProcessor(ILog log, EmotionDetector emotionDetector)
        {
            _log = log;
            _emotionDetector = emotionDetector;
            _overallEmotionData = new EmotionData();
            foreach (var prop in typeof(EmotionData).GetProperties())
            {
                prop.SetValue(_overallEmotionData, 0);
            }
        }

        private EmotionData _overallEmotionData;

        public void ProcessTweet(Tweet tweet)
        {
            if (tweet.ReTweet) return;

            var associatedEmotion = _emotionDetector.Detect(tweet.StatusMessage);

            //_log.Info($"Processing Tweet: {tweet.StatusMessage}");

            var properties = typeof(EmotionData).GetProperties();
            foreach (var prop in properties)
            {
                prop.SetValue(_overallEmotionData, (float)prop.GetValue(_overallEmotionData) + (float)prop.GetValue(associatedEmotion));
            }

            _log.Info($"Overall EmotionData: {string.Join(',', properties.Select(p => $"{p.Name}: {p.GetValue(_overallEmotionData)}").ToList())}");
        }
    }
}
