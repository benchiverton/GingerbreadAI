using System.Threading.Tasks;
using Emotion.Detector.Lexicons.Interfaces;
using log4net;
using NServiceBus;
using TweetAnalyserV1.ServiceConsole.Caches;
using TweetAnalyserV1.ServiceConsole.Persisters;
using TweetListener.Events;

namespace TweetAnalyserV1.ServiceConsole.Handlers
{
    public class TweetReceivedHandler : IHandleMessages<TweetReceived>
    {
        private readonly ILog _log;
        private readonly IEmotionDetector _emotionDetector;
        private readonly EmotionPersister _emotionPersister;
        private readonly TweetCache _tweetCache;

        public TweetReceivedHandler(ILog log, IEmotionDetector emotionDetector, EmotionPersister emotionPersister, TweetCache tweetCache)
        {
            _log = log;
            _emotionDetector = emotionDetector;
            _emotionPersister = emotionPersister;
            _tweetCache = tweetCache;
        }

        public Task Handle(TweetReceived message, IMessageHandlerContext context)
        {
            _log.Info($"Received tweet with Id: {message.TweetId}");

            if (_tweetCache.QueryContainsAndUpdateCache(message.TweetId))
            {
                _log.Debug($"Tweet with Id: {message.TweetId} has already been processed.");
                return Task.CompletedTask;
            }

            return Task.Run(() => HandleTweet(message));
        }

        private void HandleTweet(TweetReceived message)
        {
            var emotion = _emotionDetector.Detect(message.Content);

            _emotionPersister.PersistTweetEmotion(message.TweetId, emotion);
        }
    }
}
