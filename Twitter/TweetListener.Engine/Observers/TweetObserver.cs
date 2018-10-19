using CoreTweet.Streaming;
using log4net;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace TweetListener.Engine.Observers
{
    public class TweetObserver : ITweetObserver
    {
        private readonly ILog _log;
        private readonly int _interval;

        public TweetObserver(ILog log, int interval)
        {
            _log = log;
            _interval = interval;
        }

        public event Action<JObject> TweetReceived;
        public event Action StartNewObserver;

        public void OnCompleted() => _log.Info("Tweet Listener has stopped.");

        public void OnError(Exception error)
        {
            _log.Error("Your TweetObserver has crashed due to an uncaught error. Details:");
            _log.Error($"Message:\r\n{error.Message}\r\nStack trace:\r\n{error.StackTrace}");

            _log.Info("Trying to start a new Tweet Observer...");
            StartNewObserver?.Invoke();
        }

        public void OnNext(StreamingMessage value)
        {
            var tweetJson = JObject.Parse(value.Json);

            TweetReceived.Invoke(tweetJson);

            // to avoid streaming too much data (try streaming tweets related to Trump lol)
            Thread.Sleep(_interval);
        }
    }
}
