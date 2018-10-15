USE [Twitter]
GO

/****** Object:  Table [dbo].[TweetData]    Script Date: 15/10/2018 22:22:03 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[TweetData](
	[TweetId] [bigint] NOT NULL,
	[Topic] [nvarchar](50) NOT NULL,
	[Content] [nvarchar](280) NOT NULL,
	[InsertedTimeStamp] [datetime] NOT NULL,
 CONSTRAINT [PK_TweetData_1] PRIMARY KEY CLUSTERED 
(
	[TweetId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO


