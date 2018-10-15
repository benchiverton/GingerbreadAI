using Emotion.Detector.Interfaces;
using log4net;
using NServiceBus;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using TweetListener.Events;
using TwitterAnalyser.ServiceConsole.Persisters;

namespace TwitterAnalyser.ServiceConsole.Handlers
{
    public class TweetReceivedHandler : IHandleMessages<TweetReceived>
    {
        private readonly ILog _log;
        private readonly IEmotionDetector _emotionDetector;
        private readonly EmotionPersister _EmotionPersister;

        public TweetReceivedHandler(ILog log, IEmotionDetector emotionDetector, EmotionPersister emotionPersister)
        {
            _log = log;
            _emotionDetector = emotionDetector;
            _EmotionPersister = emotionPersister;
        }

        public Task Handle(TweetReceived message, IMessageHandlerContext context)
        {
            _log.Info($"Received tweet with Id: {message.TweetId}");

            return new Task(() => HandleTweet(message));
        }

        private void HandleTweet(TweetReceived message)
        {
            var emotion = _emotionDetector.Detect(message.Content);

            _EmotionPersister.PersistTweetEmotion(message.TweetId, emotion);
        }
    }
}
