using Emotion.Detector.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Emotion.Detector.Interfaces
{
    public interface IEmotionDetector
    {
        EmotionData Detect(string text);
    }
}
