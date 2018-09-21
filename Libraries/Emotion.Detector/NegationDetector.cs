namespace Emotion.Detector
{
    using System.Collections.Generic;

    public static class NegationDetector
    {
        private static readonly List<string> Negations;

        static NegationDetector()
        {
            Negations = new List<string>
            {
                "not",
                "aren't"
            };
        }

        public static bool DetectNegation(this string word)
        {
            // need to make sure that this is not case sensitive
            return Negations.Contains(word);
        }
    }
}
