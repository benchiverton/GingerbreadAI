using CoreTweet.Streaming;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace TweetListener.Engine.Observers
{
    public interface ITweetObserver : IObserver<StreamingMessage>
    {
        event Action<JObject> TweetReceived;
        event Action StartNewObserver;
    }
}
