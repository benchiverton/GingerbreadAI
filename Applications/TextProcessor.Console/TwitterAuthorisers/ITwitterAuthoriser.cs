using System;

namespace TwitterProcessor.Console.TwitterAuthorisers
{
    public interface ITwitterAuthoriser
    {
        string GetPinCode(Uri authUrl);
    }
}
