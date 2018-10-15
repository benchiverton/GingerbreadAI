USE [Twitter]
GO

/****** Object:  StoredProcedure [dbo].[PersistTweet]    Script Date: 15/10/2018 22:46:28 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
ALTER PROCEDURE [dbo].[PersistTweet] 

	@TweetId BIGINT,
	@Topic NVARCHAR(50),
	@TweetedTime DATETIME,
	@Content NVARCHAR(280)
	 
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
	INSERT INTO [dbo].[TweetData] (TweetId, Topic, Content, InsertedTimeStamp)
	VALUES (@TweetId, @Topic, @Content, GETDATE())

	INSERT INTO [dbo].[TweetHistory] (TweetId, TweetedTime, ReTweet)
	VALUES (@TweetId, @TweetedTime, 0) 
END
GO


