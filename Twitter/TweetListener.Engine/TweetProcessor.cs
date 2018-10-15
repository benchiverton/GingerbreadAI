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

        private string _topic;

        public TweetProcessor(ILog log, ITweetPersister tweetPersister, IEndpointInstance endpointInstance)
        {
            _log = log;
            _tweetPersister = tweetPersister;
            _endpointInstance = endpointInstance;
        }

        public string Topic { set => _topic = value; }

        public Task ProcessTweet(JObject tweetJson)
        {
            var process = new Task(() => Process(tweetJson));
            process.Start();
            return process;
        }

        private void Process(JObject tweetJson)
        {
            var tweetData = JsonToTweetData.Convert(tweetJson);

            _log.Info($"Tweet received. Id: {tweetData.OriginalTweetId}");
            _log.Debug($"Tweet content: {tweetData.OriginalContent}");

            _tweetPersister.PersistTweet(_topic, tweetData).GetAwaiter().GetResult();
            _log.Debug($"Successfully persisted tweet with Id: {tweetData.OriginalTweetId}");

            _endpointInstance.Publish(new TweetReceived(tweetData.OriginalTweetId, tweetData.OriginalContent)).ConfigureAwait(false);
            _log.Debug($"Successfully published 'TweetReceived' event for tweet with Id: {tweetData.OriginalTweetId}");
        }
    }
}
