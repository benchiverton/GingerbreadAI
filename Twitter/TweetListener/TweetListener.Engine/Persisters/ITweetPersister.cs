using System.Threading.Tasks;
using TweetListener.Engine.Data;

namespace TweetListener.Engine.Persisters
{
    public interface ITweetPersister
    {
        Task PersistTweet(string topic, TweetData tweetData);
    }
}
