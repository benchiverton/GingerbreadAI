USE [TextAnalysis]
GO

/****** Object:  StoredProcedure [dbo].[GetEmotionsFromWord]    Script Date: 14/10/2018 18:51:53 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE [dbo].[GetEmotionsFromWord] 
	@Word nvarchar(50)
AS
BEGIN
	SET NOCOUNT ON;

	SELECT Anger, Anticipation, Disgust, Fear, Joy, Negative, Positive, Sadness, Surprise, Trust
	FROM dbo.wordEmotions
	WHERE Word = @Word
	RETURN
END
GO


