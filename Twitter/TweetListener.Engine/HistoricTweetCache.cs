using log4net;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Text;

namespace TweetListener.Engine
{
    public class HistoricTweetCache
    {
        private readonly ILog _log;

        private List<long> _historicTweets;
        private readonly object _historicTweetsLock;

        public HistoricTweetCache(ILog log)
        {
            _log = log;
            _historicTweets = new List<long>();
            _historicTweetsLock = new object();

            var connectionString = Environment.GetEnvironmentVariable("twitterRepositoryConnectionString");
            using (var dbConnection = new SqlConnection(connectionString))
            {
                dbConnection.Open();
                SqlCommand command = new SqlCommand("[dbo].[GetHistoricTweets] ", dbConnection);
                using (var reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        _historicTweets.Add(reader.GetFieldValue<long>(0));
                    }
                }
                _log.Info($"{_historicTweets.Count} processed tweets loaded into cache!");
                dbConnection.Close();
            }
        }

        public bool QueryContainsAndUpdateCache(long tweetId)
        {
            bool isInCache = true;
            lock (_historicTweetsLock)
            {
                if (!_historicTweets.Contains(tweetId))
                {
                    isInCache = false;
                    _historicTweets.Add(tweetId);
                }
            }
            return isInCache;
        }
    }
}
