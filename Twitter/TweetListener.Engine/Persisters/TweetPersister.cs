using Dapper;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Text;
using System.Threading.Tasks;
using TweetListener.Engine.Data;

namespace TweetListener.Engine.Persisters
{
    public class TweetPersister : ITweetPersister
    {
        private readonly string _connectionString;

        public TweetPersister()
        {
            _connectionString = Environment.GetEnvironmentVariable("twitterRepositoryConnectionString");
        }

        public Task PersistTweet(string topic, TweetData tweetData)
        {
            var persist = new Task(() => Persist(topic, tweetData));
            persist.Start();
            return persist;
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
                    dbConnection.Execute("[dbo].[PersistReTweet]", spParameters, commandType: CommandType.StoredProcedure);
                }
                else
                {
                    dbConnection.Execute("[dbo].[PersistTweet]", spParameters, commandType: CommandType.StoredProcedure);
                }
            }
        }
    }
}
