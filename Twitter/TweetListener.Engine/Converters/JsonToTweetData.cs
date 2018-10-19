using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using TweetListener.Engine.Data;

namespace TweetListener.Engine.Converters
{
    public static class JsonToTweetData
    {
        private static readonly DateTime epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        public static TweetData Convert(JObject tweetJson)
        {
            if (tweetJson.TryGetValue("limit", out _))
            {
                return null; // you have made too many requests for twitter, you naughty dog
            }

            var tweet = new TweetData
            {
                TweetId = tweetJson.GetValue("id").Value<long>(),
                TweetedTime = epoch.AddMilliseconds(tweetJson.GetValue("timestamp_ms").Value<long>()),
                ReTweet = tweetJson.TryGetValue("retweeted_status", out _)
            };

            if (tweet.ReTweet)
            {
                tweetJson = tweetJson.GetValue("retweeted_status").ToObject<JObject>();
            }

            tweet.OriginalTweetId = tweetJson.GetValue("id").Value<long>();
            tweet.OriginalContent = tweetJson.GetValue("truncated").Value<bool>()
                ? tweetJson.GetValue("extended_tweet").ToObject<JObject>().GetValue("full_text").Value<string>()
                : tweetJson.GetValue("text").Value<string>();

            return tweet;
        }
    }
}
