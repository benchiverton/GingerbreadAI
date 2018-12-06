using CoreTweet.Streaming;
using log4net;
using System;
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

        public event Action<string> TweetReceived;
        public event Action ReSubscribe;

        public void OnCompleted()
        {
            _log.Info("Tweet Observer has stopped. Attempting to resubscribe...");
            ReSubscribe?.Invoke();
        }

        public void OnError(Exception error)
        {
            _log.Error("Your TweetObserver has crashed due to an uncaught error. Details:");
            _log.Error($"Message:\r\n{error.Message}\r\nStack trace:\r\n{error.StackTrace}");

            _log.Info("Trying to resubscribe...");
            ReSubscribe?.Invoke();
        }

        public void OnNext(StreamingMessage value)
        {
            TweetReceived?.Invoke(value.Json);

            // to avoid streaming too much data (try streaming tweets related to Trump lol)
            Thread.Sleep(_interval);
        }
    }
}
