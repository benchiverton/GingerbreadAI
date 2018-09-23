using CoreTweet;
using CoreTweet.Streaming;
using log4net;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using TwitterProcessor.Console.TwitterAuthorisers;

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
        private readonly IObserver<StreamingMessage> _observer;

        private string _topic;
        private Tokens _token;

        public TweetListener(ILog log, ITwitterAuthoriser twitterAuthoriser, IObserver<StreamingMessage> observer)
        {
            _consumerKey = Environment.GetEnvironmentVariable("twitterConsumerKey", EnvironmentVariableTarget.User);
            _consumerSecret = Environment.GetEnvironmentVariable("twitterConsumerSecret", EnvironmentVariableTarget.User);

            _log = log;
            _twitterAuthoriser = twitterAuthoriser;
            _observer = observer;
        }

        public void Start(string topic)
        {
            _topic = topic;

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

            _log.Info("TweetListener started!");
        }

        private void StartStreaming()
        {
            // retreives public status'
            _token.Streaming.FilterAsObservable(track: _topic).Subscribe(_observer);
        }

        private void StopStreaming()
        {
            _observer.OnCompleted();
        }
    }
}
