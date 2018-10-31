using CoreTweet.Streaming;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace TweetListener.Engine.Observers
{
    public interface ITweetObserver : IObserver<StreamingMessage>
    {
        event Action<string> TweetReceived;
        event Action<ITweetObserver> ReSubscribe;
    }
}
