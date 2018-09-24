using CoreTweet;
using CoreTweet.Streaming;
using log4net;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using TwitterProcessor.Console.TwitterAuthorisers;
using TwitterProcessor.Console.Data;
using TwitterProcessor.Console.Observers;

namespace TwitterProcessor.Console
{
    // This needs to:
    // - Start a tweet endpoint listening based on a criteria
    // - Raise an event when a tweet has been tweeted
    public class TweetListener
    {
        private readonly string _consumerKey;
        private readonly string _consumerSecret;
        private readonly ILog _log;
        private readonly ITwitterAuthoriser _twitterAuthoriser;
        private readonly ITweetObserver<StreamingMessage, Tweet> _observer;

        private string _topic;
        private Tokens _token;

        public TweetListener(ILog log, ITwitterAuthoriser twitterAuthoriser, ITweetObserver<StreamingMessage, Tweet> observer)
        {
            _consumerKey = Environment.GetEnvironmentVariable("twitterConsumerKey", EnvironmentVariableTarget.User);
            _consumerSecret = Environment.GetEnvironmentVariable("twitterConsumerSecret", EnvironmentVariableTarget.User);

            _log = log;
            _twitterAuthoriser = twitterAuthoriser;
            _observer = observer;
        }

        public void Initialise(string topic, Action<Tweet> processTweet)
        {
            _topic = topic;
            _observer.ProcessTweet += processTweet;
            _observer.StartNewObserver += StartStreaming;
        }

        public void Start()
        {
            if (_topic == null)
            {
                _log.Error("TweetListener could not be started. Please Initialise me before starting me.");
                return;
            }

            OAuth.OAuthSession session;
            try
            {
                session = OAuth.AuthorizeAsync(_consumerKey, _consumerSecret).GetAwaiter().GetResult();
                var pincode = _twitterAuthoriser.GetPinCode(session.AuthorizeUri);
                _token = session.GetTokensAsync(pincode).GetAwaiter().GetResult();
            }
            catch (Exception e)
            {
                _log.Error($"Something went wrong whilst connecting to Twitter. This TweetListener for the topic '{_topic}' was not successfully started.");
                _log.Debug($"Message:\r\n{e.Message}\r\nStack trace:\r\n{e.StackTrace}");
            }

            StartStreaming();
        }

        private void StartStreaming()
        {
            _token.Streaming.FilterAsObservable(track: _topic).Subscribe(_observer);

            _log.Info("Tweet Observer started!");
            _log.Info($"Observing tweets related to the topic '{_topic}'.");
        }

        private void StopStreaming()
        {
            _observer.OnCompleted();
        }
    }
}
