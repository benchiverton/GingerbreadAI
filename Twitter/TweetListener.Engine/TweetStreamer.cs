﻿using CoreTweet;
using CoreTweet.Streaming;
using log4net;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using TweetListener.Engine.Observers;

namespace TweetListener.Engine
{
    public class TweetStreamer
    {
        private readonly string _consumerKey;
        private readonly string _consumerSecret;
        private readonly ILog _log;
        private readonly ITweetObserver _observer;
        private readonly Tokens _token;

        private string _topic;

        public TweetStreamer(ILog log, ITweetObserver observer, Tokens token)
        {
            _consumerKey = Environment.GetEnvironmentVariable("twitterConsumerKey", EnvironmentVariableTarget.User);
            _consumerSecret = Environment.GetEnvironmentVariable("twitterConsumerSecret", EnvironmentVariableTarget.User);

            _log = log;
            _token = token;
            _observer = observer;
        }

        public void Initialise(string topic, Action<JObject> processTweet)
        {
            _topic = topic;
            _observer.TweetReceived += processTweet;
            _observer.StartNewObserver += StartStreaming;
        }

        public void Start()
        {
            if (_topic == null)
            {
                _log.Error("TweetListener could not be started. Please Initialise me before starting me.");
                return;
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