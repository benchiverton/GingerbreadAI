using Emotion.Detector;
using System;
using System.Collections.Generic;
using System.Text;

namespace TwitterProcessor.Console
{
    // This should: 
    // - manage the TweetListener
    // - manage the EmotionDetector(s)
    public class ProcessEngine
    {
        private readonly TweetListener _tweetListener;
        private readonly EmotionDetector _emotionDetector;

        public ProcessEngine(TweetListener tweetListener, EmotionDetector emotionDetector)
        {
            _tweetListener = tweetListener;
            _emotionDetector = emotionDetector;
        }

        // Start tweet listener
        // Get Events etc configured
        public void Start(string topic)
        {
            _tweetListener.Start(topic);
        }
    }
}
