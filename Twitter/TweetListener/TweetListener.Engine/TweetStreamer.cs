using CoreTweet;
using log4net;
using System;
using TweetListener.Engine.Observers;

namespace TweetListener.Engine
{
    public class TweetStreamer
    {
        private readonly ILog _log;
        private readonly ITweetObserver _observer;
        private readonly Tokens _token;

        private string _topic;
        private IDisposable _subscription;

        public TweetStreamer(ILog log, ITweetObserver observer, Tokens token)
        { 
            _log = log;
            _token = token;
            _observer = observer;
        }

        public void Initialise(string topic, Action<string> processTweet)
        {
            _topic = topic;
            _observer.TweetReceived += processTweet;
            _observer.ReSubscribe += SubScribe;
        }

        public void Start()
        {
            if (_topic == null)
            {
                _log.Error($"{this.GetType().Name} could not be started as the topic has not been assigned.");
                return;
            }

            SubScribe();
        }

        private void SubScribe()
        {
            _subscription?.Dispose();
            _subscription = _token.Streaming.FilterAsObservable(track: _topic).Subscribe(_observer);
            
            _log.Info("Tweet Observer started!");
            _log.Info($"Observing tweets related to the topic '{_topic}'.");
        }

        private void StopStreaming()
        {
            _subscription?.Dispose();
            _observer.OnCompleted();
        }
    }
}
