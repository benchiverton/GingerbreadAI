USE [Twitter]
GO

/****** Object:  Table [dbo].[TweetHistory]    Script Date: 15/10/2018 22:22:18 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[TweetHistory](
	[TweetId] [bigint] NOT NULL,
	[TweetedTime] [datetime] NOT NULL,
	[ReTweet] [bit] NOT NULL,
 CONSTRAINT [PK_TweetHistory_1] PRIMARY KEY CLUSTERED 
(
	[TweetId] ASC,
	[TweetedTime] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO


