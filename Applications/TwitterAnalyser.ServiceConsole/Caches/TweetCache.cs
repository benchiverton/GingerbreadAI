using log4net;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Text;

namespace TwitterAnalyser.ServiceConsole.Caches
{
    public class TweetCache
    {
        private readonly ILog _log;

        private List<long> _processedTweets;
        private readonly object _processedTweetsLock;

        public TweetCache(ILog log)
        {
            _log = log;
            _processedTweets = new List<long>();
            _processedTweetsLock = new object();

            var connectionString = Environment.GetEnvironmentVariable("twitterRepositoryConnectionString", EnvironmentVariableTarget.User);
            using (var dbConnection = new SqlConnection(connectionString))
            {
                dbConnection.Open();
                SqlCommand command = new SqlCommand("[dbo].[GetProcessedTweetsV1]", dbConnection);
                using (var reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        _processedTweets.Add(reader.GetFieldValue<long>(0));
                    }
                }
                _log.Info($"{_processedTweets.Count} processed tweets loaded into cache!");
                dbConnection.Close();
            }
        }

        public bool QueryContainsAndUpdateCache(long tweetId)
        {
            bool isInCache = true;
            lock (_processedTweetsLock)
            {
                if (!_processedTweets.Contains(tweetId))
                {
                    isInCache = false;
                    _processedTweets.Add(tweetId);
                }
            }
            return isInCache;
        }
    }
}
