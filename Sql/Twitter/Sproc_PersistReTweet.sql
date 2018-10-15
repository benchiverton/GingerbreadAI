USE [Twitter]
GO

/****** Object:  StoredProcedure [dbo].[PersistReTweet]    Script Date: 15/10/2018 22:27:56 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO





-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE [dbo].[PersistReTweet] 

	-- tweeted time of retweet
	@TweetedTime DATETIME,

	-- the id/topic/content of the original tweet
	@TweetId BIGINT,
	@Topic NVARCHAR(50),
	@Content NVARCHAR(280)
	 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	IF NOT EXISTS (SELECT * FROM [dbo].[TweetData] WHERE TweetId = @TweetId)
	BEGIN

	INSERT INTO [dbo].[TweetData] (TweetId, Topic, Content, InsertedTimeStamp)
	VALUES (@TweetId, @Topic, @Content, GETDATE())

	END

	INSERT INTO [dbo].[TweetHistory] (TweetId, TweetedTime, ReTweet)
	VALUES (@TweetId, @TweetedTime, 1) 
END
GO


