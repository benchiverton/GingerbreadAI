using Dapper;
using log4net;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TweetListener.Engine.Data;

namespace TweetListener.Engine.Persisters
{
    public class TweetPersister : ITweetPersister
    {
        private readonly ILog _log;
        private readonly string _connectionString;

        public TweetPersister(ILog log)
        {
            _log = log;
            _connectionString = Environment.GetEnvironmentVariable("twitterRepositoryConnectionString");
        }

        public Task PersistTweet(string topic, TweetData tweetData)
        {
            return Task.Run(() => Persist(topic, tweetData));
        }

        private void Persist(string topic, TweetData tweetData)
        {
            using (var dbConnection = new SqlConnection(_connectionString))
            {
                var spParameters = new DynamicParameters();
                spParameters.Add("@TweetId", tweetData.TweetId);
                spParameters.Add("@Topic", topic);
                spParameters.Add("@Content", tweetData.OriginalContent);
                spParameters.Add("@TweetedTime", tweetData.TweetedTime);
                if (tweetData.ReTweet)
                {
                    spParameters.Add("@OriginalTweetId", tweetData.OriginalTweetId);
                    dbConnection.Execute("[dbo].[PersistReTweet]", spParameters, commandTimeout: 30, commandType: CommandType.StoredProcedure);
                }
                else
                {
                    dbConnection.Execute("[dbo].[PersistTweet]", spParameters, commandType: CommandType.StoredProcedure);
                }
            }
        }
    }
}
