using CoreTweet.Streaming;
using log4net;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using TwitterProcessor.Console.Data;

namespace TwitterProcessor.Console.Observers
{
    public class StreamingMessageObserver : IObserver<StreamingMessage>
    {
        private readonly ILog _log;

        public StreamingMessageObserver(ILog log)
        {
            _log = log;
        }

        public void OnCompleted()
        {
            // this thingy has stopped
        }

        public void OnError(Exception error)
        {
            _log.Error("There was an issue whilst observing a streaming message. Details:");
            _log.Debug($"Message:\r\n{error.Message}\r\nStack trace:\r\n{error.StackTrace}");
        }

        public void OnNext(StreamingMessage value)
        {
            var tweetJson = JObject.Parse(value.Json);

            var tweet = new Tweet
            {
                CreatedDateTime = new DateTime(tweetJson.GetValue("timestamp_ms").Value<long>()),
                StatusMessage = tweetJson.GetValue("text").Value<string>(),
            };

            // I have something getting tweets!! Yay!!
        }
    }
}
