using Emotion.Detector.Lexicons.Data;

namespace Emotion.Detector.Lexicons.Interfaces
{
    public interface IEmotionDetector
    {
        EmotionData Detect(string text);
    }
}
