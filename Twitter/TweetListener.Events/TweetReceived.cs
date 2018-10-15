using NServiceBus;
using System;

namespace TweetListener.Events
{
    public class TweetReceived : IEvent
    {
        public TweetReceived(long tweetId, string content)
        {
            TweetId = tweetId;
            Content = content;
        }

        public long TweetId;
        public string Content;
    }
}
