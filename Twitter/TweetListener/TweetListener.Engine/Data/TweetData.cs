using System;

namespace TweetListener.Engine.Data
{
    public class TweetData
    {
        public long TweetId;
        public long OriginalTweetId;
        public string OriginalContent;
        public DateTime TweetedTime;
        public bool ReTweet;
    }
}
