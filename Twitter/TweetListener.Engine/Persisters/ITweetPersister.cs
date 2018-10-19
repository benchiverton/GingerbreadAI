using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using TweetListener.Engine.Data;

namespace TweetListener.Engine.Persisters
{
    public interface ITweetPersister
    {
        Task PersistTweet(string topic, TweetData tweetData);
    }
}
