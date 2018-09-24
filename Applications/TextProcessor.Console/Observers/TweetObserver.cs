using CoreTweet.Streaming;
using log4net;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using TwitterProcessor.Console.Data;

namespace TwitterProcessor.Console.Observers
{
    public class TweetObserver : ITweetObserver<StreamingMessage, Tweet>
    {
        private readonly ILog _log;

        public TweetObserver(ILog log)
        {
            _log = log;
        }

        public event Action<Tweet> ProcessTweet;
        public event Action StartNewObserver;

        public void OnCompleted()
        {
            _log.Info("Tweet Listener has stopped.");
        }

        public void OnError(Exception error)
        {
            _log.Error("Your TweetObserver has crashed due to an uncaught error. Details:");
            _log.Error($"Message:\r\n{error.Message}\r\nStack trace:\r\n{error.StackTrace}");

            _log.Info("Trying to start a new Tweet Observer...");
            StartNewObserver();
        }

        public void OnNext(StreamingMessage value)
        {
            var tweetJson = JObject.Parse(value.Json);

            var message = tweetJson.GetValue("truncated").Value<bool>()
                ? tweetJson.GetValue("extended_tweet").ToObject<JObject>().GetValue("full_text").Value<string>()
                : tweetJson.GetValue("text").Value<string>();
            var tweet = new Tweet
            {
                CreatedDateTime = new DateTime(tweetJson.GetValue("timestamp_ms").Value<long>()),
                Retweet = tweetJson.GetValue("retweeted").Value<bool>(),
                StatusMessage = message,
            };

            ProcessTweet(tweet);
        }
    }
}
