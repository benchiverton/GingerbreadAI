using Emotion.Detector.Interfaces;
using log4net;
using NServiceBus;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Text;
using System.Threading.Tasks;
using TweetListener.Events;
using TwitterAnalyser.ServiceConsole.Caches;
using TwitterAnalyser.ServiceConsole.Persisters;

namespace TwitterAnalyser.ServiceConsole.Handlers
{
    public class TweetReceivedHandler : IHandleMessages<TweetReceived>
    {
        private readonly ILog _log;
        private readonly IEmotionDetector _emotionDetector;
        private readonly EmotionPersister _EmotionPersister;
        private readonly TweetCache _TweetCache;

        public TweetReceivedHandler(ILog log, IEmotionDetector emotionDetector, EmotionPersister emotionPersister, TweetCache tweetCache)
        {
            _log = log;
            _emotionDetector = emotionDetector;
            _EmotionPersister = emotionPersister;
            _TweetCache = tweetCache;
        }

        public Task Handle(TweetReceived message, IMessageHandlerContext context)
        {
            _log.Info($"Received tweet with Id: {message.TweetId}");

            if (_TweetCache.QueryContainsAndUpdateCache(message.TweetId))
            {
                _log.Debug($"Tweet with Id: {message.TweetId} has already been processed.");
                return Task.CompletedTask;
            }

            return Task.Run(() => HandleTweet(message));
        }

        private void HandleTweet(TweetReceived message)
        {
            var emotion = _emotionDetector.Detect(message.Content);

            _EmotionPersister.PersistTweetEmotion(message.TweetId, emotion);
        }
    }
}
