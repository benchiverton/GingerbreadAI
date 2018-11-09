using CoreTweet.Streaming;
using System;

namespace TweetListener.Engine.Observers
{
    public interface ITweetObserver : IObserver<StreamingMessage>
    {
        event Action<string> TweetReceived;
        event Action ReSubscribe;
    }
}
