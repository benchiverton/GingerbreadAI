using System.Collections.Generic;
using System.IO;

namespace Emotion.Detector
{
    public class NegationManager
    {
        private readonly List<string> _negations;

        // All punctuation is stripped from words as we are processing them
        // Therefore punctuation should not be included in our negation list
        // Also should all be lower case
        public NegationManager(string fileLoc)
        {
            _negations = new List<string>(File.ReadAllText(fileLoc).Split("\r\n"));
        }

        public bool IsNegation(string word)
        {
            if (_negations.Contains(word))
            {
                return true;
            }

            return false;
        }
    }
}
