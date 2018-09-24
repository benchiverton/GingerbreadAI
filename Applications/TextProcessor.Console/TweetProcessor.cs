using Emotion.Detector;
using log4net;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TwitterProcessor.Console.Data;

namespace TwitterProcessor.Console
{
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
            overallEmotion = new Emotion.Detector.Data.Emotion();
            foreach (var prop in typeof(Emotion.Detector.Data.Emotion).GetProperties())
            {
                prop.SetValue(overallEmotion, 0);
            }
        }

        private Emotion.Detector.Data.Emotion overallEmotion;

        public void ProcessTweet(Tweet tweet)
        {
            if (tweet.Retweet) return;

            var associatedEmotion = _emotionDetector.Detect(tweet.StatusMessage);

            //_log.Info($"Processing Tweet: {tweet.StatusMessage}");

            var properties = typeof(Emotion.Detector.Data.Emotion).GetProperties();
            foreach (var prop in properties)
            {
                prop.SetValue(overallEmotion, (float)prop.GetValue(overallEmotion) + (float)prop.GetValue(associatedEmotion));
            }

            _log.Info($"Overall Emotion: {string.Join(',', properties.Select(p => $"{p.Name}: {p.GetValue(overallEmotion)}").ToList())}");
        }
    }
}
