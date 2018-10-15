USE [TextAnalysis]
GO

/****** Object:  Table [dbo].[word_emotions]    Script Date: 14/10/2018 17:39:30 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[wordEmotions](
	[Word] [nvarchar](50) NOT NULL,
	[Anger] [float] NOT NULL,
	[Anticipation] [float] NOT NULL,
	[Disgust] [float] NOT NULL,
	[Fear] [float] NOT NULL,
	[Joy] [float] NOT NULL,
	[Negative] [float] NOT NULL,
	[Positive] [float] NOT NULL,
	[Sadness] [float] NOT NULL,
	[Surprise] [float] NOT NULL,
	[Trust] [float] NOT NULL,
 CONSTRAINT [PK_word_emotions] PRIMARY KEY CLUSTERED 
(
	[Word] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO


