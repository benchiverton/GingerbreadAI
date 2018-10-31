using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using log4net;
using Newtonsoft.Json.Linq;
using NServiceBus;
using TweetListener.Engine.Converters;
using TweetListener.Engine.Persisters;
using TweetListener.Events;

namespace TweetListener.Engine
{
    public class TweetProcessor
    {
        private readonly ILog _log;
        private readonly ITweetPersister _tweetPersister;
        private readonly IEndpointInstance _endpointInstance;
        private readonly HistoricTweetCache _tweetCache;

        private string _topic;

        public TweetProcessor(ILog log, ITweetPersister tweetPersister, IEndpointInstance endpointInstance, HistoricTweetCache tweetCache)
        {
            _log = log;
            _tweetPersister = tweetPersister;
            _endpointInstance = endpointInstance;
            _tweetCache = tweetCache;
        }

        public string Topic { set => _topic = value; }

        public Task ProcessTweet(string tweetJson)
        {
            return Task.Run(() => Process(tweetJson));
        }

        private void Process(string tweetJson)
        {
            var tweetData = JsonToTweetData.Convert(JObject.Parse(tweetJson));

            if (tweetData == null)
            {
                _log.Warn("You have reached the limit for the tweets streamed. Streaming should continue shortly.");
                return; // we have reached twitters streaming cap
            }

            _log.Info($"Tweet received. Id: {tweetData.TweetId}");
            _log.Debug($"Tweet content: {tweetData.OriginalContent}");

            if (_tweetCache.QueryContainsAndUpdateCache(tweetData.TweetId))
            {
                _log.Warn($"Tweet with Id: {tweetData.TweetId} has already been processed.");
                return;
            }

            _tweetPersister.PersistTweet(_topic, tweetData).GetAwaiter().GetResult();
            _log.Debug($"Successfully persisted tweet with Id: {tweetData.TweetId}");

            _endpointInstance.Publish(new TweetReceived(tweetData.OriginalTweetId, tweetData.OriginalContent)).ConfigureAwait(false);
            _log.Debug($"Successfully published 'TweetReceived' event for tweet with Id: {tweetData.TweetId}");
        }
    }
}
